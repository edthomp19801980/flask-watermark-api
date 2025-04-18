from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import scipy.fftpack
import io

app = Flask(__name__)

@app.route("/")
def home():
    return "Watermark API is live."

@app.route("/extract", methods=["POST"])
def extract_watermark():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(file.stream).convert("YCbCr")
    y_channel = np.array(image.getchannel("Y"), dtype=np.float32)

    block_size = 8
    redundancy = 9
    expected_chars = 40
    bits = []

    rows, cols = y_channel.shape
    for i in range(0, rows - block_size, block_size):
        for j in range(0, cols - block_size, block_size):
            block = y_channel[i:i+block_size, j:j+block_size]
            dct_block = scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            coeff = dct_block[2, 1]
            bit = int(np.round(coeff / 10)) % 2
            bits.append(bit)
            if len(bits) >= expected_chars * 8 * redundancy:
                break
        if len(bits) >= expected_chars * 8 * redundancy:
            break

    recovered_bits = []
    for i in range(0, len(bits), redundancy):
        group = bits[i:i+redundancy]
        if len(group) == redundancy:
            voted_bit = int(np.round(np.mean(group)))
            recovered_bits.append(str(voted_bit))

    chars = []
    for i in range(0, len(recovered_bits), 8):
        byte = ''.join(recovered_bits[i:i+8])
        char_code = int(byte, 2)
        if 32 <= char_code <= 126:
            chars.append(chr(char_code))
        else:
            break

    extracted = ''.join(chars)
    return jsonify({"watermark": extracted})
