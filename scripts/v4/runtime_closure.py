"""Register runtime bytes before collection and reject unregistered lazy imports."""
from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def loaded_files():
    modules=set()
    for module in list(sys.modules.values()):
        for attribute in ('__file__','__cached__'):
            name=getattr(module,attribute,None)
            if isinstance(name,str) and Path(name).is_file():
                modules.add(Path(name).resolve())
    libraries=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        row=line.split(maxsplit=5)
        if len(row)==6 and row[5].startswith('/') and '.so' in row[5] and Path(row[5]).is_file():
            libraries.add(Path(row[5]).resolve())
    return modules,libraries


def ldconfig_libraries(output):
    """Resolve byte-valued loader paths without trimming filename characters."""
    if not isinstance(output,bytes):raise TypeError('ldconfig output must be bytes')
    libraries=set()
    for line in output.split(b'\n'):
        if b'=>' not in line:continue
        if b' => ' not in line:raise ValueError('malformed ldconfig path separator')
        raw_path=line.split(b' => ',1)[1]
        decoded=os.fsdecode(raw_path)
        if not os.path.isabs(decoded) or os.fsencode(decoded)!=raw_path:
            raise ValueError('invalid or non-roundtripping ldconfig path: '+raw_path.hex())
        if not os.path.isfile(decoded):continue
        resolved=Path(decoded).resolve(strict=True)
        try:str(resolved).encode('utf-8')
        except UnicodeEncodeError as exc:
            raise ValueError('resolved library path cannot enter UTF-8 provenance: '+os.fsencode(resolved).hex()) from exc
        libraries.add(resolved)
    return libraries


def runtime_inventory():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.markers import default_environment
    packages=metadata.packages_distributions()
    seeds={'torch','transformers','decord','flash-attn','numpy','accelerate','safetensors','pillow','einops',
           'pyarrow','matplotlib','jsonschema','evaluate','sqlitedict','datasets'}
    roots=set()
    for name,module in list(sys.modules.items()):
        seeds.update(packages.get(name.split('.')[0],()))
        filename=getattr(module,'__file__',None)
        if filename and '/site-packages/' in filename:
            base=Path(filename.split('/site-packages/')[0]+'/site-packages')
            top=base/Path(filename).relative_to(base).parts[0]
            if top.is_dir():roots.add(top)
    queue=sorted(seeds);distributions={};files={}
    environment=dict(default_environment(),extra='')
    while queue:
        name=canonicalize_name(queue.pop())
        if name in distributions:continue
        dist=metadata.distribution(name)
        distributions[name]=dist.version
        for requirement in dist.requires or ():
            req=Requirement(requirement)
            if req.marker is None or req.marker.evaluate(environment):
                queue.append(req.name)
        for item in dist.files or ():
            path=Path(dist.locate_file(item))
            if not path.is_file():continue
            if '/site-packages/' in str(path):
                relative=str(path).split('/site-packages/',1)[1]
                if '..' not in Path(relative).parts:
                    files[path.resolve()]='runtime/site-packages/'+relative
    # Imported packages without distribution metadata are still registered.
    for root in sorted(roots):
        for path in root.rglob('*'):
            if path.is_file():
                files.setdefault(path.resolve(),'runtime/site-packages/'+root.name+'/'+str(path.relative_to(root)))
    stdlib=Path(sysconfig.get_path('stdlib'))
    for current,dirs,names in os.walk(stdlib):
        dirs[:]=[d for d in dirs if d not in ('site-packages','dist-packages')]
        for name in names:
            path=Path(current)/name
            if path.is_file():files.setdefault(path.resolve(),'runtime/stdlib/'+str(path.relative_to(stdlib)))
    files[Path(sys.executable).resolve()]='runtime/python_executable'
    command=shutil.which('ldconfig') or '/sbin/ldconfig'
    linked=subprocess.run([command,'-p'],capture_output=True,check=True)
    shared=ldconfig_libraries(linked.stdout)
    for directory in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        if not directory:continue
        root=Path(directory)
        if root.is_dir():
            shared.update(p.resolve() for p in root.glob('*.so*') if p.is_file())
    modules,libraries=loaded_files()
    shared.update(libraries)
    for path in sorted(shared):
        files.setdefault(path,'runtime/system_libraries/'+path.name)
    # Explicitly include origins/caches actually imported before D3 publication.
    for path in sorted(modules):
        if '/site-packages/' in str(path):
            files.setdefault(path,'runtime/site-packages/'+str(path).split('/site-packages/',1)[1])
        elif path.is_relative_to(Path(__file__).resolve().parents[2]):
            files.setdefault(path,'runtime/observed_project/'+str(path.relative_to(Path(__file__).resolve().parents[2])))
    # Distinct library files with the same SONAME need distinct logical identities.
    by_name={}
    for path,name in sorted(files.items()):
        by_name.setdefault(name,[]).append(path)
    for name,paths in by_name.items():
        if len(paths)>1:
            for index,path in enumerate(sorted(paths)):
                files[path]=name+'/copy'+str(index)
    return files,distributions


def assert_runtime_closed(provenance):
    records={Path(x['physical_path']).resolve():x for x in provenance['execution']['files']}
    known=set(records)
    modules,libraries=loaded_files()
    missing=sorted(modules.union(libraries)-known)
    if missing:
        raise ValueError('actual runtime imports/libraries absent from D3 closure: '+repr([str(p) for p in missing]))
    bindings=[]
    for name,module in sorted(sys.modules.items()):
        for attribute in ('__file__','__cached__'):
            path=getattr(module,attribute,None)
            if isinstance(path,str) and Path(path).is_file():
                item=records[Path(path).resolve()]
                bindings.append({'module':name,'attribute':attribute,'logical_path':item['logical_path'],'sha256':item['sha256']})
    return {'loaded_module_files':sorted(str(p) for p in modules),
            'mapped_libraries':sorted(str(p) for p in libraries),'all_registered':True,
            'scientific_bindings':bindings,
            'scientific_libraries':[{'logical_path':records[p]['logical_path'],'sha256':records[p]['sha256']} for p in sorted(libraries)]}
