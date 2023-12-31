# TenSEAL Image Recognition

## Introduction

This is a simple project that uses TenSEAL for homomorphic encryption and decryption, as well as similarity calculation of images in encrypted form.

Here is the workflow of this project: 
1. The encrypted images generator encrypt the three channels of 1001 images using the BFV/CKKS scheme with single contexts for each channel. One of the images is encrypted using the client's context. 
2. Then, client retrieve all the data from SQLite and serialize the BFV/CKKS vector objects and save them in SQLite. 
3. After that, server deserialize the serialized objects and calculate the squared cosine similarity numerator and denominator between the BFV/CKKS vector of the target image and these vectors in encrypted form. 
4. Server return the numerator and denominator of all squared cosine similarities in encrypted form and client decrypt them using the client's context. 
5. Then, client calculate the squared cosine similarity for each channel separately and find the ID of the most similar image, which is returned to the database. 
6. Finally, the database retrieves the encrypted image vector corresponding to the ID and returns it to the client, who can decrypt and reconstruct the image using their private key.
- **Attention:** The "client" and "server" mentioned above are actually the same machine in my implementation, but they can be different machines in practice.

## Platform and Environment

- CPU: Intel 13700K
- GPU: NVIDIA RTX 4060 Ti 8G
- Memory: 32GB
- Operating System: Ubuntu 22.04 LTS
- Python Version: 3.10.12

## Usage

I used anime-faces dataset from huggingface. You can download the dataset from [anime-faces](https://huggingface.co/datasets/huggan/anime-faces/tree/main), unzip `data.zip` and put it in the `image` folder if you like.

Run the following command to compare the time cost of the BFV/CKKS scheme.
```bash
python3 BFV.py && python3 CKKS.py
```

## Results

I use the BFV/CKKS scheme to encrypt the three channels of 1001 images and calculate the cosine similarity between the encrypted image vectors. The results are as follows:
- `poly_modulus_degree = 8192`
- `coeff_mod_bit_sizes = [60, 40, 40, 60]`

|-|Original Image|BFV Decrypted Image|CKKS Decrypted Image|
|:---:|:---:|:---:|:---:|
|Image|![](./asset/20769.png)|![](./asset/BFV_decrypted.png)|![](./asset/CKKS_decrypted.png)|
|plain_modulus/global_scale|-|1032193|2^40|
|Cosine Similarity|-|[1.0, 1.0, 1.0]|[1.00, 0.99, 0.98]|
|Time Cost encryption|-|17.49 s|18.95 s|
|Time Cost calculation|-|227.42 s|97.89 s|

- The decrypted image is not exactly the same as the original image because the CKKS scheme use floating-point numbers to represent the encrypted data, which will cause some precision loss. But the decrypted image is still very similar to the original image.

Running time of the BFV/CKKS scheme:

![](./asset/result.png)

- As you can see, the BFV scheme offers slightly faster encryption compared to the CKKS scheme, while the CKKS scheme offers faster decryption and homomorphic operations compared to the BFV scheme.
- However, the CKKS scheme incurs some precision loss due to the use of floating-point numbers, while the BFV scheme can achieve accurate integer computations.