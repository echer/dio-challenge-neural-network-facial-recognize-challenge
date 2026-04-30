import cv2
import os

def aplicar_blur(imagem, kernel_size=5):
    """
    Aplica blur (Gaussian Blur) em uma imagem.

    Args:
        imagem: numpy array (imagem BGR ou RGB)
        kernel_size: tamanho do kernel (ímpar: 3,5,7...)

    Returns:
        imagem com blur
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # garantir ímpar

    blur = cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 0)
    return blur

def blur_img(path):
    img = cv2.imread(path)
    img_blur = aplicar_blur(img, 10)

    # salvar
    cv2.imwrite("" + path, img_blur)

blur_img("assets/detection.png")
