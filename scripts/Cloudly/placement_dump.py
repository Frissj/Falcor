"""Dump Cloudly layouts and their map references from Retoc legacy exports."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import struct

MAGIC = 0x9E2A83C1
DENSITY_OPS = {0: "Add", 1: "Subtract", 2: "HardSubtract"}
PLACEMENT_TYPES = {0: "Base", 1: "Alto", 2: "Cirrus", 3: "CirrusFar", 4: "Num", 5: "None"}
PLACEMENT_MODES = {0: "Disable", 1: "Auto"}
VOLUME_BLEND_MODES = {0: "Highest", 1: "Nearest", 2: "Lowest"}
SKY_BLEND_MODES = {0: "SmoothStep", 1: "Linear"}

# Property order and types come from the game's native reflection tables.
SHARED = (
    ("PlacementTypeOverride", "i"),
    ("PlacementMode", "i"),
    ("UserHeightOffsetInMeters", "f"),
    ("RainCloudStretch", "bool"),
    ("IsRainPillar", "bool"),
    ("RainPillarRadiusAsFraction", "f"),
    ("IsAbsorptionMask", "bool"),
    ("AbsorptionMaskRadiusAsFraction", "f"),
    ("StraightStretch", "bool"),
    ("StraightStretchStartTexcoordW", "f"),
    ("StraightStretchNumTexelsBleed", "f"),
    ("StraightStretchCoversEntireMap", "bool"),
    ("KeepDensity", "bool"),
    ("Shear", "3f"),
)
COMPONENT = (
    ("IsGroup", "bool"),
    ("Position", "4f"),
    ("Rotation", "4f"),
    ("Scale", "4f"),
    ("UVScale", "3f"),
    ("UVOffset", "3f"),
    ("NumSlices", "2i"),
    ("DensityOperation", "i"),
    ("TextureId", "i"),
    ("DensityRemap", "2f"),
    ("IsHidden", "bool"),
    ("SharedData", SHARED),
    ("UID_ForStableSorting", "I"),
    ("IsGenerated", "bool"),
    ("ComponentName", "string"),
    ("ParentId", "i"),
    ("WireframeBoxColor", "4B"),
    ("ShowWireframeBox", "bool"),
    ("WireframeBoxThickness", "f"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset

    def read(self, size):
        end = self.offset + size
        require(0 <= self.offset <= end <= len(self.data), f"Read outside payload at {self.offset:#x}")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def get(self, fmt):
        spec = struct.Struct("<" + fmt)
        values = spec.unpack(self.read(spec.size))
        require(all(not isinstance(x, float) or math.isfinite(x) for x in values), "Non-finite property")
        return values[0] if len(values) == 1 else values

    def count(self, limit=100_000):
        count = self.get("i")
        require(0 <= count <= limit, f"Invalid array count {count}")
        return count

    def string(self):
        count = self.get("i")
        if count == 0:
            return ""
        if count < 0:
            return self.read(-2 * count)[:-2].decode("utf-16-le")
        return self.read(count)[:-1].decode("utf-8")

    def boolean(self):
        value = self.get("B")
        require(value in (0, 1), f"Invalid bool {value}")
        return bool(value)

    def properties(self, limit):
        fragments = []
        end = mask_bits = 0
        while True:
            value = self.get("H")
            first = end + (value & 0x7F)
            count = value >> 9
            end = first + count
            require(end <= limit, f"Property index {end - 1} exceeds schema at {self.offset:#x}")
            masked = bool(value & 0x80)
            fragments.append((range(first, end), masked))
            if masked:
                mask_bits += count
            if value & 0x100:
                break

        mask_size = (mask_bits + 7) // 8 if mask_bits <= 16 else 4 * ((mask_bits + 31) // 32)
        mask = int.from_bytes(self.read(mask_size), "little")
        result = {}
        for indices, masked in fragments:
            for index in indices:
                result[index] = masked and bool(mask & 1)
                if masked:
                    mask >>= 1
        return result

    def record(self, schema):
        result = {}
        for index, zero in self.properties(len(schema)).items():
            name, kind = schema[index]
            if isinstance(kind, tuple):
                value = {} if zero else self.record(kind)
            elif kind == "string":
                value = "" if zero else self.string()
            elif kind == "bool":
                value = False if zero else self.boolean()
            elif zero:
                values = struct.unpack("<" + kind, bytes(struct.calcsize("<" + kind)))
                value = values[0] if len(values) == 1 else values
            else:
                value = self.get(kind)
            result[name] = value
        return result

    def end_export(self):
        require(self.read(4) == b"\0" * 4 and self.offset == len(self.data), "Unexpected UObject export tail")


class Package:
    def __init__(self, path):
        self.path = path
        r = Reader(path.read_bytes())
        require(r.get("Ii") == (MAGIC, -8), "Expected the Retoc UE5 legacy package header")
        _, _, _, _, versions = r.get("5i")
        r.read(20 * versions)
        header_size = r.get("i")
        require(header_size == len(r.data), "Package is not a separate legacy header")
        self.name = r.string()
        flags = r.get("I")
        require(flags & 0x80002000 == 0x80002000, "Expected cooked unversioned properties")
        name_count, name_offset = r.get("2i")
        r.read(16)  # soft-object path and gatherable-text table descriptors
        export_count, export_offset, import_count, import_offset = r.get("4i")

        self.names = []
        r.offset = name_offset
        for _ in range(name_count):
            self.names.append(r.string())
            r.read(4)  # name hashes

        self.imports = []
        r.offset = import_offset
        for _ in range(import_count):
            _, _, cls, cls_number, outer, name, number = r.get("7i")
            r.read(4)  # optional-import flag
            self.imports.append((self.fname(cls, cls_number), outer, self.fname(name, number)))

        self.exports = []
        r.offset = export_offset
        for _ in range(export_count):
            cls, _, _, _, name, number, _, size, offset = r.get("6iI2q")
            r.read(52)  # flags and dependency ranges for this unversioned build
            require(size >= 0 and offset >= header_size, "Invalid export range")
            self.exports.append((cls, self.fname(name, number), size, offset - header_size))

    def fname(self, index, number):
        require(0 <= index < len(self.names), f"Invalid name index {index}")
        name = self.names[index]
        return f"{name}_{number - 1}" if number else name

    def object_ref(self, index):
        if index == 0:
            return None
        if index > 0:
            return self.name + "." + self.exports[index - 1][1]
        _, outer, name = self.imports[-index - 1]
        return self.object_ref(outer) + "." + name if outer else name

    def class_name(self, index):
        if index < 0:
            return self.imports[-index - 1][2]
        return self.exports[index - 1][1] if index else None

    def payload(self, export):
        _, _, size, offset = export
        with self.path.with_suffix(".uexp").open("rb") as f:
            f.seek(offset)
            data = f.read(size)
        require(len(data) == size, f"Truncated export in {self.path}")
        return Reader(data)


def read_layout(package, export):
    r = package.payload(export)
    require(r.properties(1) == {0: False}, "Missing volume component data")
    result = {"version": None, "components": [], "textures": []}
    for index, zero in r.properties(3).items():
        if index == 0:
            result["version"] = 0 if zero else r.get("i")
        elif index == 1:
            result["components"] = [r.record(COMPONENT) for _ in range(0 if zero else r.count())]
        else:
            for _ in range(0 if zero else r.count()):
                first, first_num, second, second_num = r.get("4i")
                subpath = r.string()
                path = package.fname(first, first_num)
                name = package.fname(second, second_num)
                result["textures"].append(None if path == "None" else f"{path}.{name}" + (f":{subpath}" if subpath else ""))
    r.end_export()
    for index, component in enumerate(result["components"]):
        component["ArrayIndex"] = index
        texture_id = component["TextureId"]
        textures = result["textures"]
        component["Texture"] = textures[texture_id] if 0 <= texture_id < len(textures) else None
        component["DensityOperationName"] = DENSITY_OPS[component["DensityOperation"]]
    return result


def read_sequence(package, export):
    r = package.payload(export)
    fields = r.properties(3)
    keyframes = []
    for _ in range(0 if fields.get(0, True) else r.count(10_000)):
        keyframe = {}
        for index, zero in r.properties(7).items():
            if index == 0:
                keyframe["hiddenComponentUIDs"] = [r.get("I") for _ in range(0 if zero else r.count())]
            elif index == 1:
                keyframe["removedComponentUIDs"] = [r.get("I") for _ in range(0 if zero else r.count())]
                entries = [r.get("Ii") for _ in range(0 if zero else r.count())]
                require(len(entries) == len(dict(entries)), "Duplicate sequencer UID")
                keyframe["componentArrayPositionByUID"] = dict(entries)
            elif index == 2:
                keyframe["internalMapInitialized"] = False if zero else r.boolean()
            elif index == 3:
                keyframe["label"] = "" if zero else r.string()
            elif index == 4:
                keyframe["skyParams"] = None if zero else package.object_ref(r.get("i"))
            elif index == 5:
                keyframe["volumeComponents"] = None if zero else package.object_ref(r.get("i"))
            else:
                keyframe["time"] = 0.0 if zero else r.get("f")
        keyframes.append(keyframe)
    # The live class default object sets both enum properties to 0.
    sky = 0 if fields.get(1, True) else r.get("i")
    volume = 0 if fields.get(2, True) else r.get("i")
    r.end_export()
    return {
        "skyParamsBlendMode": SKY_BLEND_MODES[sky],
        "volumeComponentsBlendMode": VOLUME_BLEND_MODES[volume],
        "keyframes": keyframes,
    }


def add_store_references(store_index, maps_by_path, assets_by_path, map_index):
    entry_pattern = re.compile(
        r'^\S+\s+\S+\s+(\d+)\s+ExportBundleData\s+(.*?)\s+"StoreEntry .*?imported_packages: \[(.*?)\]'
    )
    reverse_imports = defaultdict(set)
    map_ids = {}
    asset_ids = {}
    with store_index.open(encoding="utf-8-sig") as f:
        for line in f:
            match = entry_pattern.match(line)
            if not match:
                continue
            package_id, path, imports = match.groups()
            path = path.removeprefix("../../../").casefold()
            if path in maps_by_path:
                map_ids[package_id] = maps_by_path[path]
            if path in assets_by_path:
                asset_ids[package_id] = assets_by_path[path]
            for dependency in re.findall(r"FPackageId\((\d+)\)", imports):
                reverse_imports[dependency].add(package_id)
    require(set(map_ids.values()) == set(map_index), "Store index does not cover every map export")
    require(len(asset_ids) == len(assets_by_path), "Store index does not cover every Cloudly asset export")

    references = defaultdict(dict)
    for package_id, assets in asset_ids.items():
        visited = {package_id}
        frontier = {package_id}
        depth = 0
        while frontier:
            for map_id in frontier & map_ids.keys():
                references[map_ids[map_id]].update(dict.fromkeys(assets, depth))
            frontier = {parent for item in frontier for parent in reverse_imports[item]} - visited
            visited.update(frontier)
            depth += 1
    for map_name, entry in map_index.items():
        entry["cloudlyStoreDependencies"] = dict(sorted(references[map_name].items()))


def dump(assets, maps, store_index=None):
    layouts = {}
    sequences = {}
    map_index = {}
    maps_by_path = {}
    assets_by_path = defaultdict(list)
    for path in sorted(set(assets.rglob("*.uasset")) | set(maps.rglob("*.umap"))):
        package = Package(path)
        root = maps if path.suffix == ".umap" else assets
        relative_path = path.relative_to(root).as_posix().casefold()
        actors = []
        embedded = []
        for export in package.exports:
            kind = package.class_name(export[0])
            ref = package.name + "." + export[1]
            if kind == "CloudlyVolumeComponents":
                layouts[ref] = read_layout(package, export)
                assets_by_path[relative_path].append(ref)
                if path.suffix == ".umap":
                    embedded.append(ref)
            elif kind == "CloudlySequencer":
                sequences[ref] = read_sequence(package, export)
                assets_by_path[relative_path].append(ref)
            elif path.suffix == ".umap" and kind and "Cloudly" in kind:
                actors.append({"object": export[1], "class": kind})
        if path.suffix == ".umap":
            entry = {
                "cloudlyExports": actors,
                "embeddedLayouts": embedded,
                "layoutImports": [],
                "sequencerImports": [],
                "skyParamsImports": [],
                "namedMapReferences": package.names,
            }
            for i, (kind, _, name) in enumerate(package.imports):
                if name.startswith("Default__"):
                    continue
                if kind == "CloudlyVolumeComponents":
                    entry["layoutImports"].append(package.object_ref(-i - 1))
                elif kind == "CloudlySequencer":
                    entry["sequencerImports"].append(package.object_ref(-i - 1))
                elif kind == "CloudlySkyParams":
                    entry["skyParamsImports"].append(package.object_ref(-i - 1))
            map_index[package.name] = entry
            maps_by_path[relative_path] = package.name

    unresolved = set()
    for map_name, entry in map_index.items():
        entry["namedMapReferences"] = sorted(set(entry["namedMapReferences"]).intersection(map_index) - {map_name})
        unresolved.update(set(entry["layoutImports"]) - set(layouts))
        unresolved.update(set(entry["sequencerImports"]) - set(sequences))
    for sequence in sequences.values():
        for keyframe in sequence["keyframes"]:
            ref = keyframe.get("volumeComponents")
            if ref and ref not in layouts:
                unresolved.add(ref)
    require(not unresolved, f"Unresolved Cloudly asset references: {sorted(unresolved)}")
    if store_index is not None:
        add_store_references(store_index, maps_by_path, assets_by_path, map_index)

    components = [c for layout in layouts.values() for c in layout["components"]]
    return {
        "formatVersion": 1,
        "propertyConventions": {
            "transforms": "Native stored Vector4f values in XYZW order. Rotation XYZ is in radians.",
            "mapReferences": "Imports and package names are static references. Runtime visibility and streaming can vary.",
            "densityOperations": DENSITY_OPS,
            "placementTypeOverrides": PLACEMENT_TYPES,
            "placementModes": PLACEMENT_MODES,
        },
        "counts": {
            "mapPackages": len(map_index),
            "mapsWithCloudlyExports": sum(bool(v["cloudlyExports"]) for v in map_index.values()),
            "layouts": len(layouts),
            "sequencers": len(sequences),
            "componentRecords": len(components),
            "groupRecords": sum(c["IsGroup"] for c in components),
            "blendOperationsExcludingGroups": dict(Counter(c["DensityOperationName"] for c in components if not c["IsGroup"])),
        },
        "maps": map_index,
        "layouts": layouts,
        "sequencers": sequences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Retoc legacy placement and sequencer exports")
    parser.add_argument("--maps", type=Path, required=True, help="Retoc legacy map exports")
    parser.add_argument("--store-index", type=Path, help="Optional output of retoc list --store")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.assets.is_dir() and args.maps.is_dir(), "Both input directories must exist")
    data = dump(args.assets, args.maps, args.store_index)
    require(data["layouts"] and data["maps"], "No layouts or maps found")
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(data["counts"], indent=2))


if __name__ == "__main__":
    main()
