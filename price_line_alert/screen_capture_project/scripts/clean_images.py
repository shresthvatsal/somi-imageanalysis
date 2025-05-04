from PIL import Image

def crop_image(input_path, output_path):
    image = Image.open(input_path)

    # Define your cropping box: (left, top, right, bottom)
    crop_box = (100, 200, 1900, 1000)  # Example; make sure right > left and bottom > top

    # Validate crop_box
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"Invalid crop_box: {crop_box}")

    cropped = image.crop(crop_box)
    cropped.save(output_path)
    print(f"✅ Cropped image saved: {output_path}")
