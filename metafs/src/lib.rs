#![warn(clippy::pedantic, clippy::nursery, rust_2018_idioms)]

use bincode::{Decode, Encode};
use bitvec::{order::Lsb0, view::BitView};
use std::{
    collections::BTreeSet,
    num::NonZeroU32,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const BINCODE_CFG: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
    bincode::config::SkipFixedArrayLength,
> = bincode::config::legacy().skip_fixed_array_length();

type Result<T> = std::result::Result<T, Error>;

type BlockNumber = u32;
type TagNumber = u16;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid checksum on {0}")]
    InvalidChecksum(String),
    #[error("no available inodes")]
    NoAvailableInode,
    #[error("no available blocks")]
    NoAvailableBlock,
    #[error("no available tags")]
    NoAvailableTag,
    #[error("optional block is none")]
    BlockIsNone,
    #[error("file of size {0} is too big (expected less than 4.26 GB)")]
    FileTooBig(usize),
    #[error(transparent)]
    SerializationError(#[from] bincode::error::EncodeError),
    #[error(transparent)]
    DeserializationError(#[from] bincode::error::DecodeError),
    #[error("{0} does not exist")]
    NoExist(&'static str),
    #[error("file name too long (len: {0}, max {1})")]
    NameTooLong(usize, usize),
}

// I like the Option<NonZeroU32> workflow more than just testing if a u32 is 0,
// however bincode encodes Option<NonZeroU32> larger than a u32
// hence, this wrapper struct
#[derive(Debug, Clone, Copy)]
pub struct OptionalU32(Option<NonZeroU32>);

impl bincode::Encode for OptionalU32 {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        match self.0 {
            Some(x) => bincode::Encode::encode(&x.get(), encoder),
            None => bincode::Encode::encode(&0, encoder),
        }
    }
}

impl bincode::Decode for OptionalU32 {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        match u32::decode(decoder) {
            // If it's zero, it will return None
            Ok(x) => Ok(Self(NonZeroU32::new(x))),
            Err(x) => Err(x),
        }
    }
}

impl From<BlockNumber> for OptionalU32 {
    fn from(val: BlockNumber) -> Self {
        Self(NonZeroU32::new(val))
    }
}

#[derive(Encode, Decode, Debug)]
pub struct Superblock {
    pub magic: u16,
    pub block_size: u32,
    pub block_count: u32,
    pub inode_count: u32,
    pub tag_count: u32,
    pub group_count: u32,
    pub created_at: u64,
    // Could actually be 0 (UNIX_EPOCH), so can't use NonZeroU64 here
    pub modified_at: Option<u64>,
    pub last_mounted_at: Option<u64>,
    pub free_blocks: u32,
    pub free_inodes: u32,
    pub free_tags: u32,
    pub blocks_per_group: u32,
    pub inodes_per_group: u32,
}

impl Superblock {
    // Random number, no significant meaning
    const MAGIC: u16 = 0xD439;

    fn new(size: u32, block_size: u32) -> Self {
        let block_count = size / block_size;
        // User blocks (block_size * 8) + Inode table blocks (block_size / 4) + block bitmap (1) + inode bitmap (1)
        let blocks_per_group = block_size * 8 + block_size / 4 + 2;
        let group_count = block_count / blocks_per_group;
        // 32 inodes in each block, block_size/2 inode blocks per group
        let inodes_per_group = 32 * (block_size / 4);
        // First group only has superblock and tags
        let inode_count = (group_count - 1) * inodes_per_group;
        // Each tag gets 1 block in the 'supergroup', minus the superblock
        let tag_count = blocks_per_group - 1;
        Self {
            magic: Self::MAGIC,
            block_size,
            block_count,
            inode_count,
            tag_count,
            group_count,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            modified_at: None,
            last_mounted_at: None,
            // block_count includes bitmaps, inode tables, and tags, only show data blocks here
            free_blocks: block_size * 8 * (group_count - 1),
            free_inodes: inode_count,
            free_tags: tag_count,
            blocks_per_group,
            inodes_per_group,
        }
    }

    fn new_from_slice(slice: &[u8]) -> Result<Self> {
        // TODO: Use dynamic block size here
        verify_checksum(slice, "superblock".to_string())?;
        Ok(bincode::decode_from_slice(&slice[4..], BINCODE_CFG)?.0)
    }
}

#[derive(Encode, Decode, Debug)]
pub struct TagTable {
    pub name: String,
    // Tag numbers range from 2-32770
    pub number: TagNumber,
    pub inodes: BTreeSet<InodeNumber>,
}

pub struct MetaFS<'a> {
    superblock: Superblock,
    raw: &'a mut [u8],
}

#[derive(Encode, Decode, Debug)]
pub struct Inode {
    pub number: InodeNumber,
    pub permissions: u16,
    pub uid: u32,
    pub gid: u32,
    // Uses superblock.block_size sized blocks, not 512 byte sized blocks
    pub block_count: u32,
    // With double indirection and 4 KiB blocks, files can only be 4.26 GB (just under u32::MAX, which could hold 4.29 GB)
    pub file_size: u32,
    pub created_at: u64,
    pub accessed_at: Option<u64>,
    pub modified_at: Option<u64>,
    pub changed_at: Option<u64>,
    pub direct_blocks: [OptionalU32; 14],
    pub indirect_block: OptionalU32,
    pub double_indirect_block: OptionalU32,
}

impl Inode {
    fn new_from_slice(slice: &[u8]) -> Result<Self> {
        verify_checksum(slice, "inode".to_string())?;
        Ok(bincode::decode_from_slice(&slice[4..128], BINCODE_CFG)?.0)
    }
}

#[derive(Encode, Decode, Debug)]
#[repr(u8)]
pub enum MultiBlock {
    FileDataWithName(String, Vec<u8>),
    FileData(Vec<u8>),
    // Block 0 is always superblock, so nothing will point to it
    Pointers(Box<[OptionalU32; 1022]>),
}

#[derive(Debug)]
struct BlockCount {
    normal: usize,
    indirect: usize,
    double_indirect: bool,
}

impl BlockCount {
    fn accumulate(&self) -> usize {
        self.normal + self.indirect + usize::from(self.double_indirect)
    }
}

#[derive(Debug, Encode, Decode, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InodeNumber(pub u32);

impl InodeNumber {
    const fn group(self, inodes_per_group: u32) -> u32 {
        (self.0 - 2) / inodes_per_group
    }

    const fn index(self, inodes_per_group: u32) -> u32 {
        (self.0 - 2) % inodes_per_group
    }
}

impl<'a> MetaFS<'a> {
    fn set_last_mounted_at(superblock: &mut Superblock, slice: &mut [u8]) -> Result<()> {
        superblock.last_mounted_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        bincode::encode_into_slice(&*superblock, &mut slice[4..], BINCODE_CFG)?;
        add_checksum(slice);
        Ok(())
    }

    pub fn new(slice: &'a mut [u8]) -> Result<Self> {
        // TODO: Use dynamic block size here
        let mut superblock = Superblock::new_from_slice(&slice[..4096])?;

        MetaFS::set_last_mounted_at(&mut superblock, &mut slice[..4096])?;

        Ok(Self {
            superblock,
            raw: slice,
        })
    }

    pub const fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    fn set_inode_bitmap(&mut self) -> Option<(u32, u32)> {
        // Skip 'supergroup'
        for i in 1..=self.superblock.group_count {
            // Start of block group + one block
            let bitmap: &mut bitvec::slice::BitSlice<u8, Lsb0> = self
                .get_slice_mut(
                    self.superblock.block_size * self.superblock.blocks_per_group * i
                        + self.superblock.block_size,
                    self.superblock.block_size,
                )
                .view_bits_mut();
            if bitmap.not_all() {
                // Guaranteed to have at least one zero
                let first_zero = bitmap.first_zero().unwrap();
                bitmap.set(first_zero, true);
                // Won't truncate, size of bitmap is less than u32::MAX
                #[allow(clippy::cast_possible_truncation)]
                return Some((first_zero as u32, i));
            }
        }
        None
    }

    // According to bincode spec:
    // 4 bytes for enum variant (per block)
    // 8 bytes for vec/string length (per block)
    // Actual vec/string data (maybe split between blocks)
    // Also 4 byte checksum (per block, not spec'd)
    // Therefore:
    // Usable block size for block with file name (vec and string): block_size - (4 + 8 + 8 + 4) = block_size - 24
    // Usable block size for block without file name (just vec): block_size - (4 + 8 + 4) = block_size - 16
    fn calc_used_inode_blocks_from_len(
        &self,
        data_len: usize,
        name_len: usize,
    ) -> Result<BlockCount> {
        let name_block_size = self.superblock.block_size as usize - 24;
        // Names longer than this can probably be supported, but not right now
        if name_len > name_block_size {
            return Err(Error::NameTooLong(name_len, name_block_size));
        }
        let normal_block_size = self.superblock.block_size as usize - 16;
        let total_len = (data_len + name_len).saturating_sub(name_block_size);
        let mut block_count = BlockCount {
            // Start with one, which will hold the name
            normal: 1,
            indirect: 0,
            double_indirect: false,
        };

        // Ceiling division
        block_count.normal += total_len / normal_block_size;
        if total_len % normal_block_size != 1 {
            block_count.normal += 1;
        }

        // Subtract direct pointers (more ceiling division)
        block_count.indirect = block_count.normal.saturating_sub(14) / 1022;
        if block_count.normal.saturating_sub(14) % 1022 != 0 {
            block_count.indirect += 1;
        }

        if block_count.indirect > 1 {
            block_count.double_indirect = true;
        }

        // Max number of blocks 1 inode can point to
        if block_count.accumulate() > 1_045_520 {
            return Err(Error::FileTooBig(total_len));
        }

        Ok(block_count)
    }

    fn get_empty_blocks(&mut self, block_count: u32) -> Result<Vec<BlockNumber>> {
        // Skip 'supergroup'
        let mut nums = Vec::with_capacity(block_count as usize);
        for _ in 0..block_count {
            nums.push(self.get_empty_block()?);
        }
        Ok(nums)
    }

    fn get_empty_block(&mut self) -> Result<BlockNumber> {
        // Skip 'supergroup'
        for i in 1..=self.superblock.group_count {
            // First block in group
            let bitmap: &mut bitvec::slice::BitSlice<u8, Lsb0> = self
                .get_slice_mut(
                    self.superblock.block_size * self.superblock.blocks_per_group * i,
                    self.superblock.block_size,
                )
                .view_bits_mut();
            if let Some(idx) = bitmap.first_zero() {
                bitmap.set(idx, true);
                // Add bitmaps and inode table
                return Ok(self.superblock.blocks_per_group * i
                    + idx as u32
                    + 2
                    + (self.superblock.block_size / 4));
            }
        }
        Err(Error::NoAvailableBlock)
    }

    fn fill_blocks(
        &mut self,
        data: &[u8],
        name: &str,
        blocks: &[BlockNumber],
        count: &BlockCount,
    ) -> Result<()> {
        let data_first = self.superblock.block_size as usize - 24 - name.len();
        let mut data_chunks =
            data[data_first.min(data.len())..].chunks(self.superblock.block_size as usize - 16);
        let mut indirect_pointers = [OptionalU32(None); 1022];
        let mut double_indirect_pointers = [OptionalU32(None); 1022];
        let mut double_indirect_blocks =
            vec![[OptionalU32(None); 1022]; count.indirect.saturating_sub(1)];
        let mut double_indirect_index = (0, 0);
        for (idx, &i) in blocks.iter().enumerate() {
            let block_slice = self.get_block_slice_from_num_mut(i);
            // Direct pointers
            if idx == 0 {
                let block_data = MultiBlock::FileDataWithName(
                    name.to_string(),
                    data[..data_first.min(data.len())].to_vec(),
                );
                bincode::encode_into_slice(block_data, &mut block_slice[4..], BINCODE_CFG)?;
            } else if idx < 14 {
                let block_data = MultiBlock::FileData(data_chunks.next().unwrap().to_vec());
                bincode::encode_into_slice(block_data, &mut block_slice[4..], BINCODE_CFG)?;
            }
            // 14 is indirect block
            // If used, 15 is double indirect block
            if count.indirect > 0 && !count.double_indirect {
                if idx > 14 {
                    indirect_pointers[idx - 15] = i.into();
                    let block_data = MultiBlock::FileData(data_chunks.next().unwrap().to_vec());
                    bincode::encode_into_slice(block_data, &mut block_slice[4..], BINCODE_CFG)?;
                }
            } else if count.indirect > 0 && count.double_indirect {
                // Fill indirect block
                if (16..1038).contains(&idx) {
                    indirect_pointers[idx - 16] = i.into();
                    let block_data = MultiBlock::FileData(data_chunks.next().unwrap().to_vec());
                    bincode::encode_into_slice(block_data, &mut block_slice[4..], BINCODE_CFG)?;
                }
                // Fill pointers to indirect blocks
                if (1038..count.indirect - 1 + 1038).contains(&idx) {
                    double_indirect_pointers[idx - 1038] = i.into();
                }
                // Fill pointed-to indirect blocks
                else if (count.indirect - 1 + 1038..count.normal - 14 + count.indirect - 1 + 1038)
                    .contains(&idx)
                {
                    double_indirect_blocks[double_indirect_index.0][double_indirect_index.1] =
                        i.into();
                    double_indirect_index.1 += 1;
                    if double_indirect_index.1 >= 1022 {
                        double_indirect_index.0 += 1;
                        double_indirect_index.1 = 0;
                    }
                }
            }
            add_checksum(block_slice);
        }
        if count.indirect > 0 {
            let block_slice = self.get_block_slice_from_num_mut(blocks[14]);
            bincode::encode_into_slice(
                MultiBlock::Pointers(Box::new(indirect_pointers)),
                &mut block_slice[4..],
                BINCODE_CFG,
            )?;
            add_checksum(block_slice);
        }
        if count.double_indirect {
            {
                let block_slice = self.get_block_slice_from_num_mut(blocks[15]);
                bincode::encode_into_slice(
                    MultiBlock::Pointers(Box::new(double_indirect_pointers)),
                    &mut block_slice[4..],
                    BINCODE_CFG,
                )?;
                add_checksum(block_slice);
            }
            for (idx, i) in double_indirect_pointers.into_iter().enumerate() {
                let i = match i.0 {
                    Some(i) => i.get(),
                    None => break,
                };
                let block_slice = self.get_block_slice_from_num_mut(i);
                bincode::encode_into_slice(
                    MultiBlock::Pointers(Box::new(double_indirect_blocks[idx])),
                    &mut block_slice[4..],
                    BINCODE_CFG,
                )?;
                add_checksum(block_slice);
            }
            for i in double_indirect_blocks {
                for j in i {
                    let j = match j.0 {
                        Some(j) => j.get(),
                        None => break,
                    };
                    let block_slice = self.get_block_slice_from_num_mut(j);
                    let block_data = MultiBlock::FileData(data_chunks.next().unwrap().to_vec());
                    bincode::encode_into_slice(block_data, &mut block_slice[4..], BINCODE_CFG)?;
                    add_checksum(block_slice);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn allocate_superblock_blocks(&mut self, count: u32) -> Result<()> {
        let superblock_slice = &mut self.raw[0..self.superblock.block_size as usize];
        verify_checksum(superblock_slice, "superblock".to_string())?;
        let mut superblock: Superblock =
            bincode::decode_from_slice(&superblock_slice[4..], BINCODE_CFG)?.0;
        superblock.free_blocks -= count;
        superblock.modified_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        bincode::encode_into_slice(superblock, &mut superblock_slice[4..], BINCODE_CFG)?;
        add_checksum(superblock_slice);
        Ok(())
    }

    #[inline]
    fn allocate_superblock_inode(&mut self) -> Result<()> {
        let superblock_slice = &mut self.raw[0..self.superblock.block_size as usize];
        verify_checksum(superblock_slice, "superblock".to_string())?;
        let mut superblock: Superblock =
            bincode::decode_from_slice(&superblock_slice[4..], BINCODE_CFG)?.0;
        superblock.free_inodes -= 1;
        superblock.modified_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        bincode::encode_into_slice(superblock, &mut superblock_slice[4..], BINCODE_CFG).unwrap();
        add_checksum(superblock_slice);
        Ok(())
    }

    #[inline]
    fn allocate_superblock_tag(&mut self) -> Result<()> {
        let superblock_slice = &mut self.raw[0..self.superblock.block_size as usize];
        verify_checksum(superblock_slice, "superblock".to_string())?;
        let mut superblock: Superblock =
            bincode::decode_from_slice(&superblock_slice[4..], BINCODE_CFG)?.0;
        superblock.free_tags -= 1;
        superblock.modified_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        bincode::encode_into_slice(superblock, &mut superblock_slice[4..], BINCODE_CFG)?;
        add_checksum(superblock_slice);
        Ok(())
    }

    #[inline]
    pub fn get_inode_from_num(&self, num: InodeNumber) -> Result<Inode> {
        if !self.check_if_inode_exists(num) {
            return Err(Error::NoExist("inode"));
        }
        let slice = self.get_inode_slice_from_num(num);
        verify_checksum(slice, format!("inode {}", num.0))?;
        Ok(bincode::decode_from_slice(&slice[4..], BINCODE_CFG)?.0)
    }

    #[inline]
    pub fn check_if_inode_exists(&self, num: InodeNumber) -> bool {
        let bitmap: &bitvec::slice::BitSlice<u8, Lsb0> = self
            .get_slice(
                num.group(self.superblock.inodes_per_group),
                self.superblock.block_size,
            )
            .view_bits();
        bitmap
            .get(num.index(self.superblock.inodes_per_group) as usize)
            .as_deref()
            .map_or(false, |x| *x)
    }

    #[inline]
    pub fn get_tag_from_num(&self, num: TagNumber) -> Result<TagTable> {
        if !self.check_if_tag_exists(num) {
            return Err(Error::NoExist("tag"));
        }
        let slice = self.get_tag_slice_from_num(num);
        verify_checksum(slice, format!("tag {}", num))?;
        Ok(bincode::decode_from_slice(&slice[4..], BINCODE_CFG)?.0)
    }

    #[inline]
    pub fn check_if_tag_exists(&self, num: TagNumber) -> bool {
        let bitmap: &bitvec::slice::BitSlice<u8, Lsb0> = self
            .get_slice(self.superblock.block_size, self.superblock.block_size)
            .view_bits();
        bitmap
            .get(num as usize - 2)
            .as_deref()
            .map_or(false, |x| *x)
    }

    #[inline]
    pub fn get_block_from_num(&self, num: BlockNumber) -> Result<MultiBlock> {
        if !self.check_if_block_exists(num) {
            return Err(Error::NoExist("block"));
        }
        let slice = self.get_block_slice_from_num(num);
        verify_checksum(slice, format!("block {}", num))?;
        Ok(bincode::decode_from_slice(&slice[4..], BINCODE_CFG)?.0)
    }

    #[inline]
    pub fn check_if_block_exists(&self, num: BlockNumber) -> bool {
        let bitmap: &bitvec::slice::BitSlice<u8, Lsb0> = self
            .get_slice(
                num / self.superblock.blocks_per_group,
                self.superblock.block_size,
            )
            .view_bits();
        bitmap
            .get((num % self.superblock.blocks_per_group) as usize)
            .as_deref()
            .map_or(false, |x| *x)
    }

    pub fn tags_iter(&self) -> impl Iterator<Item = TagTable> + '_ {
        let bitmap: &bitvec::slice::BitSlice<u8, Lsb0> = self
            .get_slice(self.superblock.block_size, self.superblock.block_size)
            .view_bits();
        bitmap
            .iter_ones()
            .filter_map(|x| self.get_tag_from_num(x as u16 + 2).ok())
    }

    #[inline]
    fn add_filename_block(&mut self, name: &str, block: BlockNumber) -> Result<()> {
        let block_slice = self.get_block_slice_from_num_mut(block);

        let block = MultiBlock::FileDataWithName(name.to_string(), Vec::new());
        bincode::encode_into_slice(block, &mut block_slice[4..], BINCODE_CFG)?;

        add_checksum(block_slice);

        Ok(())
    }

    #[inline]
    fn get_inodes_in_tags(&self, tags: &[TagNumber]) -> Result<BTreeSet<InodeNumber>> {
        let mut sets = Vec::with_capacity(tags.len());
        for &num in tags {
            let tag = self.get_tag_from_num(num)?;
            sets.push(tag.inodes);
        }
        Ok(sets.into_iter().reduce(|set1, set2| &set1 & &set2).unwrap())
    }

    fn create(&mut self, file_name: &str, tags: &[TagNumber]) -> Result<()> {
        let (_, group) = self.set_inode_bitmap().ok_or(Error::NoAvailableInode)?;

        // No inode 0 or 1 (inode and tag numbers overlap)
        let num = InodeNumber(group * self.superblock.inodes_per_group + 2);

        let mut direct_blocks = [OptionalU32(None); 14];
        let name_block = self.get_empty_block()?;
        direct_blocks[0] = OptionalU32(NonZeroU32::new(name_block));

        /*let block_count = self
            .calc_used_inode_blocks_from_len(data.len(), file_name.len())
            .ok_or(Error::FileTooBig(data.len() + file_name.len()))?;

        let block_list = self.allocate_empty_blocks(block_count.accumulate() as u32)?;

        let mut direct_blocks = [OptionalU32(None); 14];
        let mut indirect_block = OptionalU32(None);
        let mut double_indirect_block = OptionalU32(None);

        for (idx, &i) in block_list.iter().enumerate().take(16) {
            if idx < 14 {
                direct_blocks[idx] = i.into();
            }
            // If idx is 14, there has to be an indirect block
            else if idx == 14 {
                indirect_block = i.into();
            }
            // However, only put a double indirect block if required
            else if idx == 15 && block_count.double_indirect {
                double_indirect_block = i.into();
            }
        }*/

        let inode_slice = self.get_inode_slice_from_num_mut(num);

        let inode = Inode {
            number: num,
            permissions: 0o777,
            uid: 0,
            gid: 0,
            block_count: 1,
            file_size: 0,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            accessed_at: None,
            modified_at: None,
            changed_at: None,
            direct_blocks,
            indirect_block: OptionalU32(None),
            double_indirect_block: OptionalU32(None),
        };

        bincode::encode_into_slice(inode, &mut inode_slice[4..], BINCODE_CFG)?;

        add_checksum(inode_slice);

        self.allocate_superblock_inode()?;

        self.add_filename_block(file_name, name_block)?;

        for &i in tags {
            self.add_inode_to_tag(i, num)?;
        }

        Ok(())
    }

    #[inline]
    fn get_slice(&self, start: u32, size: u32) -> &[u8] {
        &self.raw[start as usize..(start + size) as usize]
    }

    #[inline]
    fn get_slice_mut(&mut self, start: u32, size: u32) -> &mut [u8] {
        &mut self.raw[start as usize..(start + size) as usize]
    }

    #[inline]
    fn get_inode_slice_from_num_mut(&mut self, num: InodeNumber) -> &mut [u8] {
        // Block group: block size * blocks per group * group inode is in
        // Add 2 blocks to account for block and inode bitmap
        // Inode: index * inode size (128)
        self.get_slice_mut(
            self.superblock.block_size
                * self.superblock.blocks_per_group
                * num.group(self.superblock.inodes_per_group)
                + self.superblock.block_size * 2
                + num.index(self.superblock.inodes_per_group) * 128,
            128,
        )
    }

    #[inline]
    fn get_inode_slice_from_num(&self, num: InodeNumber) -> &[u8] {
        self.get_slice(
            self.superblock.block_size
                * self.superblock.blocks_per_group
                * num.group(self.superblock.inodes_per_group)
                + self.superblock.block_size * 2
                + num.index(self.superblock.inodes_per_group) * 128,
            128,
        )
    }

    #[inline]
    fn get_block_slice_from_num(&self, num: BlockNumber) -> &[u8] {
        self.get_slice(num * self.superblock.block_size, self.superblock.block_size)
    }

    #[inline]
    fn get_block_slice_from_num_mut(&mut self, num: BlockNumber) -> &mut [u8] {
        self.get_slice_mut(num * self.superblock.block_size, self.superblock.block_size)
    }

    #[inline]
    fn get_tag_slice_from_num_mut(&mut self, num: TagNumber) -> &mut [u8] {
        self.get_slice_mut(
            self.superblock.block_size * (num as u32 - 2) + self.superblock.block_size * 2,
            self.superblock.block_size,
        )
    }

    #[inline]
    fn get_tag_slice_from_num(&self, num: TagNumber) -> &[u8] {
        self.get_slice(
            self.superblock.block_size * (num as u32 - 2) + self.superblock.block_size * 2,
            self.superblock.block_size,
        )
    }

    fn get_block_range_from_offset_size(
        &self,
        offset: u32,
        size: u32,
        file_name_len: u32,
    ) -> [(u32, u32); 2] {
        let name_block_len = self.superblock.block_size - 24 - file_name_len;
        let normal_block_len = self.superblock.block_size - 16;
    }

    pub fn read_data(&self, offset: u32, size: u32, inode: &Inode) -> Result<Vec<u8>> {
        let mut file_data: Vec<u8> = Vec::with_capacity(inode.file_size as usize);

        for i in inode.direct_blocks {
            let i = match i.0 {
                Some(i) => i.get(),
                None => break,
            };
            let block_slice = self.get_block_slice_from_num(i);
            verify_checksum(block_slice, format!("data block {}", i))?;
            let data: MultiBlock = bincode::decode_from_slice(&block_slice[4..], BINCODE_CFG)?.0;
            match data {
                MultiBlock::FileData(mut block_data)
                | MultiBlock::FileDataWithName(_, mut block_data) => {
                    file_data.append(&mut block_data)
                }
                // Direct pointers should never point to a pointer block
                MultiBlock::Pointers(_) => unreachable!(),
            }
        }

        if let Some(indirect_block) = inode.indirect_block.0 {
            let indirect_block = indirect_block.get();

            let pointers = {
                let data = self.get_block_from_num(indirect_block)?;

                match data {
                    // Should always be a pointer block
                    MultiBlock::Pointers(block_pointers) => *block_pointers,
                    _ => unreachable!(),
                }
            };

            for i in pointers {
                let i = match i.0 {
                    Some(i) => i.get(),
                    None => break,
                };

                let data = self.get_block_from_num(i)?;

                match data {
                    // Should always be a normal data block
                    MultiBlock::FileData(mut block_data) => file_data.append(&mut block_data),
                    _ => unreachable!(),
                }
            }
        }

        if let Some(double_indirect_block) = inode.double_indirect_block.0 {
            let double_indirect_block = double_indirect_block.get();

            let indirect_block_pointers = {
                let data = self.get_block_from_num(double_indirect_block)?;

                match data {
                    // Should always be a pointer block
                    MultiBlock::Pointers(block_pointers) => *block_pointers,
                    _ => unreachable!(),
                }
            };

            for i in indirect_block_pointers {
                let i = match i.0 {
                    Some(i) => i.get(),
                    None => break,
                };

                let pointers = {
                    let data = self.get_block_from_num(i)?;

                    match data {
                        // Should always be a pointer block
                        MultiBlock::Pointers(block_pointers) => *block_pointers,
                        _ => unreachable!(),
                    }
                };

                for j in pointers {
                    let j = match j.0 {
                        Some(j) => j.get(),
                        None => break,
                    };

                    let data = self.get_block_from_num(j)?;

                    match data {
                        // Should always be a normal data block
                        MultiBlock::FileData(mut block_data) => file_data.append(&mut block_data),
                        _ => unreachable!(),
                    }
                }
            }
        }

        Ok(file_data)
    }

    pub fn read_name(&self, inode: &Inode) -> Result<String> {
        // There needs to be at least one direct block
        let block_num = inode.direct_blocks[0].0.unwrap().get();

        let block_slice = self.get_block_slice_from_num(block_num);

        verify_checksum(block_slice, format!("data block {}", block_num))?;

        let data: MultiBlock = bincode::decode_from_slice(&block_slice[4..], BINCODE_CFG)?.0;

        Ok(match data {
            // Should always be a pointer block
            MultiBlock::FileDataWithName(file_name, _) => file_name,
            _ => unreachable!(),
        })
    }

    fn create_tag(&mut self, name: &str) -> Result<u16> {
        // Starts after superblock
        let bitmap: &mut bitvec::slice::BitSlice<u8, Lsb0> = self
            .get_slice_mut(self.superblock.block_size, self.superblock.block_size)
            .view_bits_mut();
        if let Some(first_zero) = bitmap.first_zero() {
            bitmap.set(first_zero, true);
            // Skip first 2 blocks
            #[allow(clippy::cast_possible_truncation)]
            let slice = self.get_slice_mut(
                self.superblock.block_size * (first_zero as u32 + 2),
                self.superblock.block_size,
            );
            #[allow(clippy::cast_possible_truncation)]
            let tag = TagTable {
                name: name.to_string(),
                inodes: BTreeSet::new(),
                // No tag 0
                number: first_zero as u16 + 2,
            };
            bincode::encode_into_slice(tag, &mut slice[4..], BINCODE_CFG)?;
            add_checksum(slice);
            Ok(first_zero as u16 + 2)
        } else {
            Err(Error::NoAvailableInode)
        }
    }

    fn add_inode_to_tag(&mut self, tag_num: u16, inode_num: InodeNumber) -> Result<()> {
        let tag_slice = self.get_tag_slice_from_num_mut(tag_num);
        verify_checksum(tag_slice, format!("tag {}", tag_num))?;

        let mut tag: TagTable = bincode::decode_from_slice(&tag_slice[4..], BINCODE_CFG)?.0;

        tag.inodes.insert(inode_num);

        bincode::encode_into_slice(tag, &mut tag_slice[4..], BINCODE_CFG)?;

        add_checksum(tag_slice);

        Ok(())
    }

    fn remove_inode_from_tag(&mut self, tag_num: u16, inode_num: InodeNumber) -> Result<()> {
        let tag_slice = self.get_tag_slice_from_num_mut(tag_num);
        verify_checksum(tag_slice, format!("tag {}", tag_num))?;

        let mut tag: TagTable = bincode::decode_from_slice(&tag_slice[4..], BINCODE_CFG)?.0;

        tag.inodes.remove(&inode_num);

        bincode::encode_into_slice(tag, &mut tag_slice[4..], BINCODE_CFG)?;

        add_checksum(tag_slice);

        Ok(())
    }
}

fn add_checksum(slice: &mut [u8]) {
    let checksum = crc32fast::hash(&slice[4..]);
    slice[..4].copy_from_slice(checksum.to_le_bytes().as_slice());
}

fn verify_checksum(slice: &[u8], name: String) -> Result<()> {
    let checksum = crc32fast::hash(&slice[4..]);
    if slice[..4] != checksum.to_le_bytes() {
        return Err(Error::InvalidChecksum(name));
    }
    Ok(())
}
