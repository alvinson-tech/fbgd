import os

DATASET_PATH = 'dataset'

print("Checking dataset structure...\n")
print(f"Looking in: {os.path.abspath(DATASET_PATH)}\n")

if not os.path.exists(DATASET_PATH):
    print("❌ ERROR: 'dataset' folder not found!")
    print("Please create a 'dataset' folder in your project directory.")
    exit()

folders = os.listdir(DATASET_PATH)
print(f"Found {len(folders)} folders/files in dataset:\n")

total_images = 0

for item in folders:
    item_path = os.path.join(DATASET_PATH, item)
    
    if os.path.isdir(item_path):
        files = os.listdir(item_path)
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        all_images = [f for f in files if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        
        print(f"📁 {item}/")
        print(f"   Total files: {len(files)}")
        print(f"   .bmp files: {len(bmp_files)}")
        print(f"   All image files: {len(all_images)}")
        
        if len(all_images) > 0:
            print(f"   Sample files: {all_images[:3]}")
        
        total_images += len(bmp_files)
        print()
    else:
        print(f"⚠️  {item} (not a folder)")
        print()

print(f"\n{'='*50}")
print(f"Total .bmp images found: {total_images}")
print(f"{'='*50}\n")

if total_images == 0:
    print("❌ No .bmp images found!")
    print("\nPossible issues:")
    print("1. Dataset is in the wrong location")
    print("2. Images are in a nested folder structure")
    print("3. Images have different extensions (.jpg, .png, etc.)")
    print("\nPlease check your dataset structure.")
else:
    print("✅ Dataset looks good! You can proceed with training.")