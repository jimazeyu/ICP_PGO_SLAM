from PIL import Image
import os

dataset = 21

print("Generating gif...")
print("About 10 seconds")
image_folder = f'./output/map_combined_{dataset}'
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

gif_images = []
for image in images:
    gif_images.append(Image.open(os.path.join(image_folder, image)))

gif_images[0].save(f'./assets/map_combined_{dataset}.gif', save_all=True, append_images=gif_images[1:], optimize=False, duration=100, loop=0)