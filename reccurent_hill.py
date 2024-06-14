import numpy as np
from gmpy2 import invert
import math

text = 'DARKSOULS'
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
key1 = np.array([[1, 8, 6],
                 [5, 9, 9],
                 [4, 5, 10]])

key2 = np.array([[3, 6, 7],
                 [9, 13, 15],
                 [2, 5, 7]])
#
# key_inv1 = np.array([[11, 8 ,20],
#                     [22, 22, 19],
#                     [21, 17, 19]])
# key_inv2 = np.array([[22, 5, 23],
#                      [5, 21, 2],
#                      [5, 17, 7]])
# key_inv3 = np.array([[3, 1, 10],
#                      [13, 16, 17],
#                      [4, 13, 10]])
#
# print(np.dot(key_inv2, np.array([2, 14, 0])) % 26)

block_size = len(key1)

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


blocks = split_into_blocks(text, block_size, alphabet)


def encode(blocks, key1, key2, alphabet):
    length = len(alphabet)
    encode_array = []
    cipher_text = ''
    if int(math.gcd(round(np.linalg.det(key1)), length)) and int(math.gcd(round(np.linalg.det(key2)), length)) != 1:
        raise ValueError('НОД определителя ключа и мощности алфавита должен равняться 1.')
    else:
        y1 = np.dot(key1, blocks[0])
        encode_array.append(y1)
        if len(blocks) > 1:
            y2 = np.dot(key2, blocks[1])
            encode_array.append(y2)
            for i in range(2, len(blocks)):
                key3 = np.dot(key1, key2) % length
                encode_array.append(np.dot(key3, blocks[i]))
                key2, key1 = key3, key2
        for i in range(len(blocks)):
            for j in range(len(blocks[i])):
                index_char = encode_array[i][j] % len(alphabet)
                encode_array[i][j] = index_char
        for index in encode_array:
            for element in index:
                if element < len(alphabet):
                    cipher_text += alphabet[element]
    return cipher_text


def to_inv_matrix(matrix, modulo):
    matrix = matrix % modulo
    det = int(round(np.linalg.det(matrix))) % modulo
    inv_det_mod = invert(det, modulo)
    inv = (inv_det_mod * (np.linalg.det(matrix)) * np.linalg.inv(matrix))
    for i in range(len(inv)):
        for j in range(len(inv[i])):
            inv[i, j] = (inv[i, j] % modulo)
    return inv

en_message = encode(blocks, key1, key2, alphabet)

blocks_encoded = split_into_blocks(en_message, block_size, alphabet)


def decode(key1, key2, alphabet, blocks_encoded):
    length = len(alphabet)
    decode_array = [None] * len(blocks_encoded)
    decode_array = np.array(decode_array)
    text = ''
    key_inv1 = to_inv_matrix(key1, length)
    key_inv2 = to_inv_matrix(key2, length)
    decode_array[0] = np.dot(key_inv1, blocks_encoded[0])
    decode_array[1] = np.dot(key_inv2, blocks_encoded[1])
    for i in range(2, len(blocks_encoded)):
        key = np.dot(key1, key2) % length
        inv_key = to_inv_matrix(key, length)
        decode_array[i] = np.dot(inv_key, blocks_encoded[i])
        key2, key1 = key, key2
    decode_array = np.vstack(decode_array)
    for i in range(len(decode_array)):
        for j in range(len(decode_array[i])):
            decode_array[i, j] = int(round(decode_array[i, j])) % length
    for index in decode_array:
        for element in index:
            if element < len(alphabet):
                text += alphabet[element]
    return text


en_message = encode(blocks, key1, key2, alphabet)
print(en_message)

de_message = decode(key1, key2, alphabet, blocks_encoded)
print(de_message.replace('$', ''))
