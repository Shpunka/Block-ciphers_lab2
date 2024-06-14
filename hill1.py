import numpy as np
import math
from gmpy2 import invert

# key = np.array([[6, 24, 1],
#                 [13, 16, 10],
#                 [19, 17, 15]])

key = np.array([[5, 1],
                [3, 2]])


def text_to_indices(text, alphabet):
    indices = []
    for char in text:
        if char in alphabet:
            indices.append(alphabet.index(char))
    return indices


def split_into_blocks(text, block_size, alphabet):
    indices = text_to_indices(text, alphabet)

    while len(indices) % block_size != 0:
        indices.append(0)

    blocks = [indices[i:i + block_size] for i in range(0, len(indices), block_size)]
    return blocks


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
text = "CITY"
text = text.upper()
block_size = len(key)


blocks = split_into_blocks(text, block_size, alphabet)


def encode(blocks, key, alphabet):
    length = len(alphabet)
    encode_message = []
    cipher_text = ''
    # if math.gcd(int(np.linalg.det(key)), length) != 1:
    #     raise ValueError('НОД определителя ключа и мощности алфавита должен равняться 1.')
    # else:
    for i in range(len(blocks)):
        encode_message.append(np.dot(key, blocks[i]))
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            index_char = encode_message[i][j] % len(alphabet)
            encode_message[i][j] = index_char
    encode_message = np.vstack(encode_message)
    # print(encode_message)
    for index in encode_message:
        for element in index:
            if element < len(alphabet):
                cipher_text += alphabet[element]
    return cipher_text


def inv_key(matrix, alphabet):
    det = int(round(np.linalg.det(matrix)))
    inv_det_mod = invert(det, len(alphabet))
    inv_matrix = (inv_det_mod * (np.linalg.det(matrix)) * np.linalg.inv(matrix))
    for i in range(len(inv_matrix)):
        for j in range(len(inv_matrix[i])):
            inv_matrix[i, j] = int(round((inv_matrix[i, j] % len(alphabet))))
    return inv_matrix


key_invert = inv_key(key, alphabet)

encode_message = encode(blocks, key, alphabet)
print("Encode message:", encode_message)

blocks_encoded = split_into_blocks(encode_message, block_size, alphabet)


def decode(key_inv, alphabet, blocks_encoded):
    lenght = len(alphabet)
    decode_array = []
    text = ''
    for i in range(len(blocks_encoded)):
        decode_array.append(np.dot(key_inv, blocks_encoded[i]))
    for i in range(len(blocks_encoded)):
        for j in range(len(blocks[i])):
            index_char = decode_array[i][j] % len(alphabet)
            decode_array[i][j] = index_char

    decode_array = np.vstack(decode_array)

    for index in decode_array:
        for element in index:
            if element < len(alphabet):
                text += alphabet[element]
    return text


de_message = decode(key_invert, alphabet, blocks_encoded)
de_message = de_message.replace('$', '')

print("Decode message:", de_message)
