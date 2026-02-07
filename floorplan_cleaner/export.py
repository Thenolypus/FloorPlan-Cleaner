import os
import json
import shutil
import numpy as np
from PIL import Image

from .models import Room, ApartmentUnit
from .mask_generator import MaskGenerator


class Exporter:
    def export_all(
        self,
        svg_path: str,
        input_name: str,
        rooms: list[Room],
        units: list[ApartmentUnit],
        mask_generator: MaskGenerator,
        output_dir: str,
    ):
        # Create INPUT_NAME subfolder
        base_dir = os.path.join(output_dir, input_name)
        os.makedirs(base_dir, exist_ok=True)

        metadata = {
            "source_svg": input_name + ".svg",
            "units": [],
        }

        for unit in units:
            unit_dir = os.path.join(base_dir, f"unit_{unit.id}")
            os.makedirs(unit_dir, exist_ok=True)

            unit_rooms = [r for r in rooms if r.unit_id == unit.id]

            unit_entry = {
                "unit_id": unit.id,
                "rooms": [],
            }

            for room_idx, room in enumerate(unit_rooms, start=1):
                room_type = room.label or "unlabelled"
                mask = mask_generator.generate_mask(room)

                filename = f"unit_{unit.id}_room_{room_idx}_{room_type}.png"
                filepath = os.path.join(unit_dir, filename)
                relative_path = f"unit_{unit.id}/{filename}"

                img = Image.fromarray(mask, mode="L")
                img.save(filepath)

                print(f"Saved: {filepath} | unique values: {np.unique(mask).tolist()}")

                room_entry = {
                    "room_id": room_idx,
                    "room_type": room_type,
                    "bbox_in_svg": {
                        "x": round(room.bbox_svg[0], 2),
                        "y": round(room.bbox_svg[1], 2),
                        "width": round(room.bbox_svg[2], 2),
                        "height": round(room.bbox_svg[3], 2),
                    },
                    "output_file": relative_path,
                }
                unit_entry["rooms"].append(room_entry)

            # Generate combined mask for the unit
            combined_mask = mask_generator.generate_combined_mask(unit_rooms)
            combined_filename = f"unit_{unit.id}_combined.png"
            combined_path = os.path.join(unit_dir, combined_filename)
            Image.fromarray(combined_mask, mode="L").save(combined_path)
            print(f"Saved: {combined_path}")

            unit_entry["combined_file"] = f"unit_{unit.id}/{combined_filename}"
            metadata["units"].append(unit_entry)

        # Copy preprocessed SVG to output root
        svg_dest = os.path.join(base_dir, input_name + "_centered.svg")
        shutil.copy2(svg_path, svg_dest)
        metadata["centered_svg"] = input_name + "_centered.svg"
        print(f"Saved: {svg_dest}")

        # Write metadata
        meta_path = os.path.join(base_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved: {meta_path}")