import os
from src.cache_generator.cache_builder import CacheBuilder

def main():
    # Get project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    squad_path = os.path.abspath(os.path.join(base_dir, "..", "data", "train-v1.1.json"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "data", "cache_dataset.csv"))

    builder = CacheBuilder(
        squad_path=squad_path,
        output_path=output_path
    )

    builder.build(limit=100000)


if __name__ == "__main__":
    main()