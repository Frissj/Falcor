"""Recover placement DXIL and its embedded source from a Retoc SM6 shader archive."""

import argparse
import mmap
from pathlib import Path
import re
import struct
import subprocess


PLACEMENT_INDICES = (13, 279, 280, 281, 282, 283, 284)


def extract(archive, output, dxc):
    with archive.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        version, = struct.unpack_from("<I", data)
        if version != 2:
            raise ValueError(f"Expected shader archive version 2, found {version}")
        offset = 4
        sections = []
        for stride in (20, 20, 16, 17, 16, 4):
            count, = struct.unpack_from("<I", data, offset)
            offset += 4
            end = offset + count * stride
            sections.append(data[offset:end])
            offset = end
        _, shader_hashes, _, shader_entries, _, _ = sections

        output.mkdir(parents=True, exist_ok=True)
        for index in PLACEMENT_INDICES:
            entry = struct.Struct("<QIIB")
            shader_offset, size, raw_size, frequency = entry.unpack_from(shader_entries, index * entry.size)
            if frequency != 5 or size != raw_size:
                raise ValueError(f"Shader {index} is not an uncompressed compute shader")
            start = offset + shader_offset
            end = start + size
            container = data.find(b"DXBC", start, end)
            if container < 0 or container + 32 > end:
                raise ValueError(f"Shader {index} is missing its DXIL container")
            container_size, = struct.unpack_from("<I", data, container + 24)
            if container + container_size > end:
                raise ValueError(f"Shader {index} has a truncated DXIL container")
            shader_hash = shader_hashes[index * 20:(index + 1) * 20].hex()
            path = output / f"{index:04d}-{shader_hash}.dxil"
            path.write_bytes(data[container:container + container_size])
            disassembly = subprocess.run([dxc, "-dumpbin", str(path)], check=True, capture_output=True).stdout
            path.with_suffix(".ll").write_bytes(disassembly)
            metadata = re.search(rb'^!dx\.source\.contents = !\{!(\d+)\}', disassembly, re.M)
            if metadata is None:
                raise ValueError(f"Shader {index} is missing embedded source metadata")
            source = re.search(rb'^!' + metadata[1] + rb' = !\{!"[^"]+", !"(.*)"\}\r?$', disassembly, re.M)
            if source is None:
                raise ValueError(f"Shader {index} is missing its embedded source record")
            decoded = re.sub(rb'\\([0-9a-fA-F]{2})', lambda match: bytes.fromhex(match[1].decode("ascii")), source[1])
            path.with_suffix(".hlsl").write_bytes(decoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dxc", default="dxc.exe")
    args = parser.parse_args()
    extract(args.archive, args.output, args.dxc)


if __name__ == "__main__":
    main()
