import tenseal as ts
import sqlite3
import numpy as np
from PIL import Image
import time

data_path = "./image/data/"
db_path = 'encrypted_color_data_bfv.db'

def save_to_db(encrypted_r, encrypted_g, encrypted_b, db_path, number):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS encrypted_color_images (
            id INTEGER PRIMARY KEY,
            encrypted_r BLOB,
            encrypted_g BLOB,
            encrypted_b BLOB
        )
    ''')

    # 检查对应id是否已存在
    cursor.execute('SELECT COUNT(*) FROM encrypted_color_images WHERE id = ?', (number,))
    if cursor.fetchone()[0] == 1:
        # 若已存在，则更新
        cursor.execute('UPDATE encrypted_color_images SET encrypted_r = ?, encrypted_g = ?, encrypted_b = ? WHERE id = ?',
                    (encrypted_r.serialize(), encrypted_g.serialize(), encrypted_b.serialize(), number))
    else:
        # 若不存在，则插入
        cursor.execute('INSERT OR IGNORE INTO encrypted_color_images (id, encrypted_r, encrypted_g, encrypted_b) VALUES (?, ?, ?, ?)',
                    (number, encrypted_r.serialize(), encrypted_g.serialize(), encrypted_b.serialize()))
    conn.commit()

    # 关闭连接
    conn.close()

def load_from_db(db_path, number):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 从数据库中获取加密数据
    cursor.execute('SELECT id, encrypted_r, encrypted_g, encrypted_b FROM encrypted_color_images WHERE id = ?', (number,))
    id, encrypted_r_blob, encrypted_g_blob, encrypted_b_blob = cursor.fetchone()

    # 关闭连接
    conn.close()

    return id, encrypted_r_blob, encrypted_g_blob, encrypted_b_blob

def encrypt(image_path, context):
    # 加载彩色图片
    image = Image.open(image_path)
    image = image.resize((64, 64))  # 调整大小为64x64

    # 转为NumPy数组，分别获取每个通道的数据
    image_array = np.array(image)
    r_channel, g_channel, b_channel = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]

    # 对每个通道进行加密
    encrypted_r = ts.bfv_vector(context, r_channel.flatten().tolist())
    encrypted_g = ts.bfv_vector(context, g_channel.flatten().tolist())
    encrypted_b = ts.bfv_vector(context, b_channel.flatten().tolist())

    return encrypted_r, encrypted_g, encrypted_b

def decrypt(encrypted_r_blob, encrypted_g_blob, encrypted_b_blob, save_path, context):
    # 加载加密数据
    encrypted_r = ts.bfv_vector_from(context, encrypted_r_blob)
    encrypted_g = ts.bfv_vector_from(context, encrypted_g_blob)
    encrypted_b = ts.bfv_vector_from(context, encrypted_b_blob)

    # 解密数据
    decrypted_r = encrypted_r.decrypt()
    decrypted_g = encrypted_g.decrypt()
    decrypted_b = encrypted_b.decrypt()

    # 将解密后的数据还原为彩色图像
    decrypted_image_array = np.zeros((64, 64, 3), dtype=np.uint8)
    decrypted_image_array[:,:,0] = np.array(decrypted_r).reshape(64, 64)
    decrypted_image_array[:,:,1] = np.array(decrypted_g).reshape(64, 64)
    decrypted_image_array[:,:,2] = np.array(decrypted_b).reshape(64, 64)

    # 显示解密后的图像
    decrypted_image = Image.fromarray(decrypted_image_array)

    # 将解密后的图像保存到文件
    decrypted_image.save(save_path)

def cosine_similarity(encrypted_1, encrypted_2):
    # 计算两个密文的平方余弦相似度
    # 分子
    numerator = encrypted_1.dot(encrypted_2)
    numerator = numerator * numerator
    # 分母
    denominator = encrypted_1.dot(encrypted_1) * encrypted_2.dot(encrypted_2)
    return numerator, denominator

def cal_similarity(numerator, denominator):
    numerator_array = np.array(numerator)
    denominator_array = np.array(denominator)
    similarity = np.sum(numerator_array) / np.sum(denominator_array)
    
    return similarity

def cal_all_similarity(encrypted_r, encrypted_g, encrypted_b, target_r, target_g, target_b):
    numerator_r, denominator_r = cosine_similarity(encrypted_r, target_r)
    numerator_g, denominator_g = cosine_similarity(encrypted_g, target_g)
    numerator_b, denominator_b = cosine_similarity(encrypted_b, target_b)

    return numerator_r, numerator_g, numerator_b, denominator_r, denominator_g, denominator_b

def decrypt_and_cal_similarity(numerator_r, numerator_g, numerator_b, denominator_r, denominator_g, denominator_b, context):
    numerator_r = numerator_r.decrypt(secret_key=context.secret_key())
    numerator_g = numerator_g.decrypt(secret_key=context.secret_key())
    numerator_b = numerator_b.decrypt(secret_key=context.secret_key())
    denominator_r = denominator_r.decrypt(secret_key=context.secret_key())
    denominator_g = denominator_g.decrypt(secret_key=context.secret_key())
    denominator_b = denominator_b.decrypt(secret_key=context.secret_key())

    similarity_r = cal_similarity(numerator_r, denominator_r)
    similarity_g = cal_similarity(numerator_g, denominator_g)
    similarity_b = cal_similarity(numerator_b, denominator_b)

    similarity = [similarity_r, similarity_g, similarity_b]
    
    return similarity

def batch_generate_encrypted_image(data_path, db_path, start_number, end_number):
    for i in range(start_number, end_number):
        image_path = data_path + str(i) + ".png"

        context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
        context.global_scale = 2**20
        context.generate_galois_keys()

        encrypted_r, encrypted_g, encrypted_b = encrypt(image_path, context)
        save_to_db(encrypted_r, encrypted_g, encrypted_b, db_path, i)
        print("\r", "[!] Progress: ", i, end="")
    print("\n [+] Batch generate encrypted image done.")
    return

def cal_all_similarity_from_db(db_path, context, save_path, target_r, target_g, target_b):
    similarity_list = []
    id_list = []
    # 全表扫描
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, encrypted_r, encrypted_g, encrypted_b FROM encrypted_color_images')
    for id, encrypted_r_blob, encrypted_g_blob, encrypted_b_blob in cursor.fetchall():
        # 加载加密数据
        encrypted_r = ts.bfv_vector_from(context, encrypted_r_blob)
        encrypted_g = ts.bfv_vector_from(context, encrypted_g_blob)
        encrypted_b = ts.bfv_vector_from(context, encrypted_b_blob)

        # 计算相似度
        numerator_r, numerator_g, numerator_b, denominator_r, denominator_g, denominator_b = cal_all_similarity(encrypted_r, encrypted_g, encrypted_b, target_r, target_g, target_b)
        similarity = decrypt_and_cal_similarity(numerator_r, numerator_g, numerator_b, denominator_r, denominator_g, denominator_b, context)
        similarity_list.append(similarity)
        id_list.append(id)

        # 进度条
        print("\r", "[!] Progress: ", id, end="")
    conn.close()

    # 找出相似度最接近1的图像
    most_similarity = []
    for i in similarity_list:
        # 计算i与[1, 1, 1]的欧氏距离
        distance = np.linalg.norm(np.array(i) - np.array([1, 1, 1]))
        # 找出最接近[1, 1, 1]的图像
        if len(most_similarity) == 0:
            most_similarity = i
            most_similarity_distance = distance
        else:
            if distance < most_similarity_distance:
                most_similarity = i
                most_similarity_distance = distance

    most_similarity_index = similarity_list.index(most_similarity)
    most_similarity_id = id_list[most_similarity_index]
    print("\n [+] Most similarity id: ", most_similarity_id)
    print(" [+] Most similarity: ", most_similarity)
    # 保存最相似的图像
    id, encrypted_r_blob, encrypted_g_blob, encrypted_b_blob = load_from_db(db_path, most_similarity_id)
    decrypt(encrypted_r_blob, encrypted_g_blob, encrypted_b_blob, save_path, context)

    print("\n [+] Calculate all similarity from db done.")
    return

if __name__ == "__main__":
    print("BFV start...")
    context_1 = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
    context_1.generate_galois_keys()

    image_path = "./image/data/20769.png"
    # 另存一份公开的context
    context_public = context_1.copy()
    context_public.make_context_public()
    # 加密
    encrypted_r, encrypted_g, encrypted_b = encrypt(image_path, context_public)
    save_to_db(encrypted_r, encrypted_g, encrypted_b, db_path, 1001)

    # 批量生成加密图像
    start_number = 1
    end_number = 1000
    save_path = "./BFV_decrypted.png"

    time_start = time.time()
    batch_generate_encrypted_image(data_path, db_path, start_number, end_number + 1)
    time_end = time.time()
    print(' [+] Encrypt 1000 images cost: {:.2f}s\n'.format(time_end - time_start))

    time_start = time.time()
    cal_all_similarity_from_db(db_path, context_1, save_path, encrypted_r, encrypted_g, encrypted_b)
    time_end = time.time()
    print(' [+] Calculate 1001 similarity cost: {:.2f}s\n'.format(time_end - time_start))