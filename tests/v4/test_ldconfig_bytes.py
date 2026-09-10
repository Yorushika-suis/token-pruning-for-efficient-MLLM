"""Loader aliases with non-UTF-8 bytes and trailing tabs must not lose targets."""
import os
from pathlib import Path
import pytest
from v4 import runtime_closure as C


def row(path):
    return b'\tfixture (libc6,x86-64) => '+os.fsencode(path)+b'\n'


def test_non_utf8_alias_and_trailing_tab_register_real_target(tmp_path):
    target=tmp_path/'libfixture.so';target.write_bytes(b'library bytes')
    alias=os.fsencode(tmp_path)+b'/\x04\xc4\t'
    os.symlink(os.fsencode(target),alias)
    assert C.ldconfig_libraries(b'1 libs in cache\n'+row(alias))=={target.resolve()}
    assert os.readlink(alias)==os.fsencode(target)
    assert target.read_bytes()==b'library bytes'


def test_utf8_whitespace_filenames_and_duplicate_aliases(tmp_path):
    first=tmp_path/'library \t';first.write_bytes(b'a')
    second=tmp_path/'\u5e93.so';second.write_bytes(b'b')
    alias=tmp_path/'alias.so';alias.symlink_to(first)
    assert C.ldconfig_libraries(row(first)+row(second)+row(alias))=={first.resolve(),second.resolve()}


def test_non_utf8_regular_target_fails_instead_of_silent_omission(tmp_path):
    path=os.fsencode(tmp_path)+b'/\xc4.so'
    with open(path,'wb') as f:f.write(b'real library')
    with pytest.raises(ValueError,match='cannot enter UTF-8 provenance'):
        C.ldconfig_libraries(row(path))


@pytest.mark.parametrize('value,error',[(b'x => relative.so\n',ValueError),
                                      (b'x=>/absolute.so\n',ValueError),
                                      ('decoded text',TypeError)])
def test_malformed_path_and_text_are_rejected(value,error):
    with pytest.raises(error):C.ldconfig_libraries(value)


def test_stale_cache_entries_remain_non_inputs(tmp_path):
    missing=tmp_path/'missing.so'
    assert C.ldconfig_libraries(row(missing))==set()


def test_cache_directory_entry_with_trailing_slash_is_not_a_file(tmp_path):
    assert C.ldconfig_libraries(row(os.fsencode(tmp_path)+b'/'))==set()


def test_redundant_separator_does_not_drop_a_real_library(tmp_path):
    target=tmp_path/'libfixture.so';target.write_bytes(b'library')
    assert C.ldconfig_libraries(row(os.fsencode(tmp_path)+b'//libfixture.so'))=={target.resolve()}
