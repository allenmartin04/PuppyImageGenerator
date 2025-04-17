# Install dependencies
!pip install diffusers transformers torch accelerate flask pyngrok flask-cors requests pillow -q

# Import libraries
import gc
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
import base64
from io import BytesIO
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The Dog API key
DOG_API_KEY = "YOUR DOG API KEY"
BASE_API_URL = "https://api.thedogapi.com/v1/images/search"
FALLBACK_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/512px-YellowLabradorLooking_new.jpg"

# Load dog-specific ViT model
try:
    processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
    model = AutoModelForImageClassification.from_pretrained(
        "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit",
        token="YOUR HUGGING FACE TOKEN"
    )
    model = model.to("cuda")
    logging.info("Dog-specific ViT model loaded successfully")
except Exception as e:
    logging.error(f"Error loading ViT model: {str(e)}")
    raise

# Load SD 1.5 pipelines once globally
try:
    text_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    logging.info("SD 1.5 pipelines loaded successfully")
except Exception as e:
    logging.error(f"Error loading SD 1.5 pipelines: {str(e)}")
    raise

# 50 breeds with traits (expanded from your list)
breed_traits = {
    "labrador_retriever": "short thick fur, floppy ears, sturdy build",
    "rottweiler": "short black-tan fur, muscular frame, semi-erect ears",
    "german_shepherd": "tan-black fur, muscular frame, erect ears",
    "golden_retriever": "silky golden fur, floppy ears, bushy tail",
    "beagle": "tricolor fur, droopy ears, short legs",
    "bulldog": "wrinkled skin, fawn fur, flat face",
    "poodle": "curly fur, rounded head, drop ears",
    "pug": "fawn fur, wrinkled face, curled tail",
    "affenpinscher": "wiry coarse black or gray fur, round head, small ears",
    "afghan_hound": "long silky cream fur, narrow head, droopy ears",
    "airedale_terrier": "wiry tan-black fur, square build, folded ears",
    "akita": "thick white fur, curled tail, bear-like face",
    "alaskan_malamute": "fluffy gray-white fur, bushy tail, muscular build",
    "american_bulldog": "short white or brindle fur, broad chest, square head",
    "american_eskimo_dog": "fluffy white fur, curled tail, pointed ears",
    "australian_cattle_dog": "blue or red speckled fur, compact body, pointed ears",
    "australian_shepherd": "wavy black-white fur, bushy tail, semi-erect ears",
    "bichon_frise": "curly white fur, round face, dark eyes",
    "border_collie": "black-white fur, lean body, semi-erect ears",
    "boston_terrier": "black-white fur, square build, bat-like ears",
    "boxer": "fawn fur with white, athletic frame, folded ears",
    "bullmastiff": "fawn or red fur, massive build, droopy jowls",
    "cavalier_king_charles_spaniel": "silky ruby-red fur, round head, floppy ears",
    "chihuahua": "fawn or black fur, tiny frame, large ears",
    "chinese_crested": "hairless with tufted fur, slender legs, erect ears",
    "cocker_spaniel": "silky black fur, floppy ears, elegant body",
    "collie": "flowing sable fur, narrow snout, feathered tail",
    "dachshund": "black-tan fur, long body, short legs",
    "dalmatian": "white fur with black spots, sleek build, rounded ears",
    "doberman_pinscher": "sleek black-tan fur, lean body, erect ears",
    "english_springer_spaniel": "liver-white fur, floppy ears, feathered legs",
    "great_dane": "smooth fawn fur, tall body, floppy ears",
    "greyhound": "fawn or brindle fur, slender build, small ears",
    "havanese": "silky white fur, sturdy body, drop ears",
    "husky": "fluffy black-white fur, erect ears, blue eyes",
    "jack_russell_terrier": "white-brown fur, sturdy body, V-shaped ears",
    "lhasa_apso": "gold fur with black tips, short legs, lion mane",
    "maltese": "silky white fur, tiny frame, drop ears",
    "miniature_pinscher": "shiny black-tan fur, elegant body, erect ears",
    "newfoundland": "shaggy black-brown fur, massive build, floppy ears",
    "papillon": "fringed white-black fur, butterfly ears, delicate frame",
    "pekingese": "gold fur, flat face, short legs",
    "pit_bull": "brindle or blue fur, broad chest, square head",
    "saint_bernard": "shaggy white-red fur, broad body, droopy jowls",
    "samoyed": "fluffy white fur, curled tail, upturned mouth",
    "shiba_inu": "red fur with markings, curled tail, fox-like face",
    "shih_tzu": "flowing white-gold fur, flat face, drop ears",
    "siberian_husky": "fluffy black-white fur, erect ears, multi-colored eyes",
    "weimaraner": "silver-gray fur, lean build, amber eyes"
}

# Breed mapping for ViT quirks
vit_breed_mapping = {
    "maltese_dog": "maltese",
    "shih": "shih_tzu",
    "golden_retriever": "golden_retriever"
}

# Breed to The Dog API breed ID mapping (approximate)
breed_to_id = {
    "labrador_retriever": "34", "rottweiler": "44", "german_shepherd": "27", "golden_retriever": "28",
    "beagle": "10", "bulldog": "16", "poodle": "42", "pug": "43", "affenpinscher": "1", "afghan_hound": "2",
    "airedale_terrier": "3", "akita": "4", "alaskan_malamute": "5", "american_bulldog": "6",
    "american_eskimo_dog": "7", "australian_cattle_dog": "8", "australian_shepherd": "9",
    "bichon_frise": "12", "border_collie": "13", "boston_terrier": "14", "boxer": "15",
    "bullmastiff": "17", "cavalier_king_charles_spaniel": "18", "chihuahua": "19", "chinese_crested": "20",
    "cocker_spaniel": "21", "collie": "22", "dachshund": "23", "dalmatian": "24", "doberman_pinscher": "25",
    "english_springer_spaniel": "26", "great_dane": "29", "greyhound": "30", "havanese": "31",
    "husky": "32", "jack_russell_terrier": "33", "lhasa_apso": "35", "maltese": "36",
    "miniature_pinscher": "37", "newfoundland": "38", "papillon": "39", "pekingese": "40",
    "pit_bull": "41", "saint_bernard": "45", "samoyed": "46", "shiba_inu": "47", "shih_tzu": "48",
    "siberian_husky": "49", "weimaraner": "50"
}

def fetch_breed_image(breed):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    if DOG_API_KEY:
        headers["x-api-key"] = DOG_API_KEY

    breed_normalized = breed.strip().lower().replace(" ", "_")
    breed_id = breed_to_id.get(breed_normalized, "34")  # Default to labrador if not found
    url = f"{BASE_API_URL}?breed_id={breed_id}&size=med&limit=1"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries} to fetch image for {breed} from {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0 and data[0].get("url"):
                img_response = requests.get(data[0]["url"], headers=headers, timeout=10, stream=True)
                img_response.raise_for_status()
                img = Image.open(io.BytesIO(img_response.content)).convert("RGB").resize((512, 512))
                detected_breed = recognize_breed(img.resize((224, 224)))
                mapped_breed = vit_breed_mapping.get(detected_breed, detected_breed)
                if mapped_breed == breed_normalized or mapped_breed in breed_normalized:
                    logging.info(f"Verified: {breed} matches detected/mapped breed **{mapped_breed}**")
                    return img
                else:
                    logging.warning(f"Mismatch: {breed} image detected as **{mapped_breed}**, retrying")
                    raise Exception("Breed mismatch")
            else:
                logging.warning(f"API returned no image for {breed}, attempting next try")
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.RequestException as e:
            logging.error(f"Fetch attempt {attempt + 1} failed for {breed} from {url}: {str(e)}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"Processing error for {breed} on attempt {attempt + 1}: {str(e)}")
            time.sleep(2 ** attempt)

    # Fallback to static image
    try:
        logging.info(f"Falling back to default image for {breed} from {FALLBACK_URL}")
        fallback_response = requests.get(FALLBACK_URL, headers=headers, timeout=10, stream=True)
        fallback_response.raise_for_status()
        img = Image.open(io.BytesIO(fallback_response.content)).convert("RGB").resize((512, 512))
        logging.info(f"Successfully fetched fallback image for {breed}")
        return img
    except Exception as e:
        logging.error(f"Failed to fetch fallback image for {breed}: {str(e)}")
        return None

def decode_base64_image(base64_string):
    try:
        if not base64_string:
            raise ValueError("Empty base64 string")
        if 'data:image' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data)).convert("RGB").resize((224, 224))
        logging.info(f"Base64 image decoded, size={len(base64_string)} bytes, dimensions={img.size}")
        return img
    except Exception as e:
        logging.error(f"Error decoding base64 image: {str(e)}")
        return None
    finally:
        gc.collect()

def recognize_breed(image):
    if image is None:
        logging.error("No valid image for breed recognition")
        return "mixed"
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=-1)
        confidence = confidence.item()
        predicted_class_idx = predicted_class_idx.item()
        predicted_label = model.config.id2label[predicted_class_idx]
        breed = predicted_label.lower().replace(" ", "_").replace("standard_", "").replace("toy_", "").replace("miniature_", "")
        if breed not in breed_traits:
            logging.warning(f"Detected breed {breed} not in supported list, defaulting to mixed")
            breed = "mixed"
        logging.info(f"Recognized: **{breed}** (Confidence: {confidence:.2f}, Label: {predicted_label})")
        if confidence < 0.05:
            logging.warning(f"Low confidence ({confidence:.2f}) for **{breed}**, defaulting to mixed")
            return "mixed"
        return breed
    except Exception as e:
        logging.error(f"Error recognizing breed: {str(e)}")
        return "mixed"
    finally:
        torch.cuda.empty_cache()
        gc.collect()

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/generate_puppy', methods=['POST'])
def generate_puppy():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        logging.info(f"Received request: {data}")
        mode = data.get('mode', 'text')

        # Immediate response with loading message
        if request.args.get('init') == 'true':
            return jsonify({
                "status": "generating",
                "message": "Generating your puppy images... This may take ~60 seconds. Please wait.",
                "note": "Note: Generation may take ~60 seconds due to high-quality rendering."
            })

        if mode == 'image':
            img1_base64 = data.get('image1')
            img2_base64 = data.get('image2')
            if not img1_base64 or not img2_base64:
                return jsonify({"error": "Both images required"}), 400

            img1 = decode_base64_image(img1_base64)
            img2 = decode_base64_image(img2_base64)
            if not img1 or not img2:
                return jsonify({"error": "Failed to process images"}), 400

            breed1 = recognize_breed(img1)
            breed2 = recognize_breed(img2)
            logging.info(f"Recognized breeds: **{breed1}**, **{breed2}**")
        else:
            breed1 = data.get('breed1', 'mixed').lower().replace(" ", "_").replace("_", " ")
            breed2 = data.get('breed2', 'mixed').lower().replace(" ", "_").replace("_", " ")

        traits1 = breed_traits.get(breed1.replace(" ", "_"), "mixed features").split(", ")
        traits2 = breed_traits.get(breed2.replace(" ", "_"), "mixed features").split(", ")
        trait1_1 = traits1[0].strip()
        trait1_2 = traits1[1].strip() if len(traits1) > 1 else "generic trait"
        trait2_1 = traits2[0].strip()
        trait2_2 = traits2[1].strip() if len(traits2) > 1 else "generic trait"

        img1 = fetch_breed_image(breed1.replace(" ", "_"))
        img2 = fetch_breed_image(breed2.replace(" ", "_"))
        logging.info(f"img1: type={type(img1)}, img2: type={type(img2)}")
        init_image_available = img1 is not None and img2 is not None and isinstance(img1, Image.Image) and isinstance(img2, Image.Image)
        logging.info(f"Init image available: {init_image_available}")

        prompts = [
            f"Photorealistic {breed1}-{breed2} puppy, 25% {breed1}, 75% {breed2}, muscular {breed2} build, full body, {trait2_1}, {trait2_2}, slight {trait1_1}, grassy park, 8K DSLR, entire dog.",
            f"Photorealistic {breed1}-{breed2} puppy, 50% {breed1}, 50% {breed2}, full body, {trait1_1} from {breed1}, {trait2_1} from {breed2}, hybrid ears, vibrant outdoor, 8K DSLR, entire dog.",
            f"Photorealistic {breed1}-{breed2} puppy, 75% {breed1}, 25% {breed2}, agile {breed1} build, full body, {trait1_1}, subtle {trait2_1}, sunny field, 8K DSLR, entire dog."
        ]

        negative_prompt = "blurry, low-res, cropped, partial body, cartoonish, distorted"

        image_urls = []
        ratios = [
            f"25% {breed1} and 75% {breed2}",
            f"50% {breed1} and 50% {breed2}",
            f"75% {breed1} and 25% {breed2}"
        ]
        for ratio, prompt in zip(ratios, prompts):
            try:
                logging.info(f"Generating {ratio} image: {prompt}")
                if init_image_available:
                    pipe = img2img_pipe
                    alpha = 0.25 if ratio.startswith("25%") else 0.5 if "50%" in ratio else 0.75
                    blended_image = Image.blend(img1, img2, alpha)
                    if not isinstance(blended_image, Image.Image):
                        logging.error(f"Blended image invalid for {ratio}, resizing to 512x512")
                        blended_image = blended_image.resize((512, 512), Image.LANCZOS)
                    logging.info(f"Blended image: type={type(blended_image)}, size={blended_image.size}")
                    with torch.no_grad():
                        image = pipe(
                            prompt,
                            init_image=blended_image,
                            strength=0.7,
                            num_inference_steps=150,
                            guidance_scale=20.0,
                            negative_prompt=negative_prompt
                        ).images[0]
                else:
                    pipe = text_pipe
                    logging.warning(f"No valid init images for {breed1}-{breed2}, using text mode")
                    with torch.no_grad():
                        image = pipe(
                            prompt,
                            num_inference_steps=150,
                            guidance_scale=20.0,
                            height=512,
                            width=512,
                            negative_prompt=negative_prompt
                        ).images[0]

                image_path = f"/tmp/puppy_{breed1.replace(' ', '_')}_{breed2.replace(' ', '_')}_{ratio.replace(' ', '_').replace('%', '')}.png"
                image.save(image_path)
                if os.path.exists(image_path):
                    public_url = f"{ngrok_url}/image/{image_path.split('/')[-1]}"
                    image_urls.append({"ratio": ratio, "image_url": public_url})
                    logging.info(f"Generated {ratio} image at: {public_url}")
                else:
                    logging.error(f"Image not saved at {image_path}")
            except Exception as e:
                logging.error(f"Error generating {ratio} image: {str(e)}")
                try:
                    pipe = text_pipe
                    with torch.no_grad():
                        image = pipe(
                            prompt,
                            num_inference_steps=150,
                            guidance_scale=20.0,
                            height=512,
                            width=512,
                            negative_prompt=negative_prompt
                        ).images[0]
                    image_path = f"/tmp/puppy_{breed1.replace(' ', '_')}_{breed2.replace(' ', '_')}_{ratio.replace(' ', '_').replace('%', '')}_fallback.png"
                    image.save(image_path)
                    if os.path.exists(image_path):
                        public_url = f"{ngrok_url}/image/{image_path.split('/')[-1]}"
                        image_urls.append({"ratio": ratio, "image_url": public_url})
                        logging.info(f"Fallback generated {ratio} image at: {public_url}")
                    else:
                        logging.error(f"Fallback image not saved at {image_path}")
                except Exception as e2:
                    logging.error(f"Fallback failed for {ratio}: {str(e2)}")

        if not image_urls:
            return jsonify({"error": "No images generated"}), 500

        response = {
            "images": image_urls,
            "recognized_breed1": f"<b>{breed1}</b>" if mode == 'image' else "N/A",
            "recognized_breed2": f"<b>{breed2}</b>" if mode == 'image' else "N/A"
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        gc.collect()

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Start ngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR NGROK TOKEN")
public_tunnel = ngrok.connect(5000)
ngrok_url = public_tunnel.public_url
print(f"Public URL: {ngrok_url}")

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)