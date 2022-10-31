use std::time::{Duration, UNIX_EPOCH};

use fuser::{FileAttr, MountOption};
use metafs::{Inode, InodeNumber, MetaFS, TagTable};

const TTL: Duration = Duration::from_secs(1);

struct MetaFSFuse<'a> {
    inner: MetaFS<'a>,
    handle: u64,
}

impl<'a> MetaFSFuse<'a> {
    fn new(file: &'a mut memmap2::MmapMut) -> Self {
        Self {
            inner: MetaFS::new(file).unwrap(),
            handle: 0,
        }
    }
}

impl<'a> fuser::Filesystem for MetaFSFuse<'a> {
    fn getattr(&mut self, _req: &fuser::Request<'_>, ino: u64, reply: fuser::ReplyAttr) {
        let attr = if ino < 32770 {
            if ino == 1 || self.inner.check_if_tag_exists(ino as u16) {
                FileAttr {
                    ino,
                    size: 0,
                    blocks: 1,
                    atime: UNIX_EPOCH,
                    mtime: UNIX_EPOCH,
                    ctime: UNIX_EPOCH,
                    crtime: UNIX_EPOCH,
                    kind: fuser::FileType::Directory,
                    perm: 0o777,
                    // Tag is used in multiple places, but only exists once
                    nlink: 1,
                    uid: 0,
                    gid: 0,
                    rdev: 0,
                    blksize: self.inner.superblock().block_size,
                    flags: 0,
                }
            } else {
                reply.error(libc::ENOENT);
                return;
            }
        } else {
            let inode = match self.inner.get_inode_from_num(InodeNumber(ino as u32)) {
                Ok(inode) => inode,
                Err(e) => {
                    reply.error(match e {
                        metafs::Error::NoExist(_) => libc::ENOENT,
                        metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                        metafs::Error::DeserializationError(_) => libc::EIO,
                        _ => unreachable!(),
                    });
                    return;
                }
            };
            FileAttr {
                ino,
                size: inode.file_size.into(),
                blocks: inode.block_count.into(),
                atime: UNIX_EPOCH + Duration::from_secs(inode.accessed_at.unwrap_or(0)),
                mtime: UNIX_EPOCH + Duration::from_secs(inode.modified_at.unwrap_or(0)),
                ctime: UNIX_EPOCH + Duration::from_secs(inode.changed_at.unwrap_or(0)),
                crtime: UNIX_EPOCH + Duration::from_secs(inode.created_at),
                kind: fuser::FileType::RegularFile,
                perm: inode.permissions,
                nlink: 1,
                uid: inode.uid,
                gid: inode.gid,
                rdev: 0,
                blksize: self.inner.superblock().block_size,
                flags: 0,
            }
        };
        reply.attr(&TTL, &attr);
    }

    fn lookup(
        &mut self,
        _req: &fuser::Request<'_>,
        parent: u64,
        name: &std::ffi::OsStr,
        reply: fuser::ReplyEntry,
    ) {
        if parent == 1 {
            for tag in self.inner.tags_iter() {
                if tag.name.as_str() == name {
                    reply.entry(
                        &TTL,
                        &FileAttr {
                            ino: tag.number.into(),
                            size: 0,
                            blocks: 0,
                            atime: UNIX_EPOCH,
                            mtime: UNIX_EPOCH,
                            ctime: UNIX_EPOCH,
                            crtime: UNIX_EPOCH,
                            kind: fuser::FileType::Directory,
                            perm: 0o777,
                            nlink: 1,
                            uid: 0,
                            gid: 0,
                            rdev: 0,
                            blksize: self.inner.superblock().block_size,
                            flags: 0,
                        },
                        0,
                    );
                    return;
                }
            }
            reply.error(libc::ENOENT);
            return;
        } else if (2..32770).contains(&parent) {
            let tag = match self.inner.get_tag_from_num(parent as u16) {
                Ok(tag) => tag,
                Err(e) => {
                    reply.error(match e {
                        metafs::Error::NoExist(_) => libc::ENOENT,
                        metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                        metafs::Error::DeserializationError(_) => libc::EIO,
                        _ => unreachable!(),
                    });
                    return;
                }
            };

            for &i in &tag.inodes {
                // Should inode names be stored in the tag?
                let inode = match self.inner.get_inode_from_num(i) {
                    Ok(inode) => inode,
                    Err(e) => {
                        reply.error(match e {
                            metafs::Error::NoExist(_) => libc::ENOENT,
                            metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                            metafs::Error::DeserializationError(_) => libc::EIO,
                            _ => unreachable!(),
                        });
                        return;
                    }
                };
                let file_name = self.inner.read_name(&inode).unwrap();
                if file_name.as_str() == name {
                    reply.entry(
                        &TTL,
                        &FileAttr {
                            ino: inode.number.0.into(),
                            size: inode.file_size.into(),
                            blocks: inode.block_count.into(),
                            atime: UNIX_EPOCH + Duration::from_secs(inode.accessed_at.unwrap_or(0)),
                            mtime: UNIX_EPOCH + Duration::from_secs(inode.modified_at.unwrap_or(0)),
                            ctime: UNIX_EPOCH + Duration::from_secs(inode.changed_at.unwrap_or(0)),
                            crtime: UNIX_EPOCH + Duration::from_secs(inode.created_at),
                            kind: fuser::FileType::RegularFile,
                            perm: inode.permissions,
                            nlink: 1,
                            uid: inode.uid,
                            gid: inode.gid,
                            rdev: 0,
                            blksize: self.inner.superblock().block_size,
                            flags: 0,
                        },
                        0,
                    );
                    return;
                }
            }
            reply.error(libc::ENOENT);
            return;
        }

        // Parent isn't a directory
        reply.error(libc::ENOTDIR);
    }

    fn readdir(
        &mut self,
        _req: &fuser::Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: fuser::ReplyDirectory,
    ) {
        if offset == 0 {
            // Inode 1 only lists tags
            if ino == 1 {
                if reply.add(1, 0, fuser::FileType::Directory, ".")
                    || reply.add(1, 1, fuser::FileType::Directory, "..")
                {
                    reply.error(libc::ENOBUFS);
                    return;
                }
                for (idx, tag) in self.inner.tags_iter().enumerate() {
                    if reply.add(
                        tag.number.into(),
                        idx as i64 + 2,
                        fuser::FileType::Directory,
                        &tag.name,
                    ) {
                        reply.error(libc::ENOBUFS);
                        return;
                    }
                }
            } else if (2..32770).contains(&ino) {
                let tag = match self.inner.get_tag_from_num(ino as u16) {
                    Ok(tag) => tag,
                    Err(e) => {
                        reply.error(match e {
                            metafs::Error::NoExist(_) => libc::ENOENT,
                            metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                            metafs::Error::DeserializationError(_) => libc::EIO,
                            _ => unreachable!(),
                        });
                        return;
                    }
                };

                if reply.add(tag.number.into(), 0, fuser::FileType::Directory, ".")
                    || reply.add(ino, 1, fuser::FileType::Directory, "..")
                {
                    reply.error(libc::ENOBUFS);
                    return;
                }
                for (idx, &i) in tag.inodes.iter().enumerate() {
                    // Should inode names be stored in the tag?
                    let inode = match self.inner.get_inode_from_num(i) {
                        Ok(inode) => inode,
                        Err(e) => {
                            reply.error(match e {
                                metafs::Error::NoExist(_) => libc::ENOENT,
                                metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                                metafs::Error::DeserializationError(_) => libc::EIO,
                                _ => unreachable!(),
                            });
                            return;
                        }
                    };
                    let file_name = self.inner.read_name(&inode).unwrap();
                    if reply.add(
                        i.0.into(),
                        idx as i64 + 2,
                        fuser::FileType::RegularFile,
                        file_name,
                    ) {
                        reply.error(libc::ENOBUFS);
                        return;
                    }
                }
            } else {
                reply.error(libc::ENOTDIR);
                return;
            }
        }
        reply.ok();
    }

    fn read(
        &mut self,
        _req: &fuser::Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: fuser::ReplyData,
    ) {
        let inode = match self.inner.get_inode_from_num(InodeNumber(ino as u32)) {
            Ok(inode) => inode,
            Err(e) => {
                reply.error(match e {
                    metafs::Error::NoExist(_) => libc::ENOENT,
                    metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                    metafs::Error::DeserializationError(_) => libc::EIO,
                    _ => unreachable!(),
                });
                return;
            }
        };

        // Protect against RequestTooBig errors
        if offset as u32 + size > inode.file_size {
            size = inode.file_size - offset as u32
        }

        let data = match self.inner.read_data(offset as u32, size, &inode) {
            Ok(data) => data,
            Err(e) => {
                reply.error(match e {
                    metafs::Error::InvalidChecksum(_) => libc::ESTALE,
                    metafs::Error::DeserializationError(_) => libc::EIO,
                    _ => unreachable!(),
                });
                return;
            }
        };

        reply.data(&data);
    }
}

fn main() {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/var/home/mprice/metatest.img")
        .expect("couldn't open file");

    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).expect("couldn't memory map file") };

    let mut fs = MetaFSFuse::new(&mut mmap);

    fs.inner.add_inode_to_tag(2, InodeNumber(32770));

    dbg!(fs.inner.tags_iter().collect::<Vec<TagTable>>());

    fuser::mount2(fs, "/mnt/test", &[MountOption::RO]).expect("mounting fs failed");

    //let inode = fs.inner.get_inode_from_num(InodeNumber(32772)).unwrap();

    //println!("reading data");

    //dbg!(std::str::from_utf8(&fs.inner.read_data(4721605, 6, &inode).unwrap()).unwrap());
}
