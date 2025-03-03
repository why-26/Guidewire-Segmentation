import numpy as np
import imageio

def dice_coefficient(true_label, prediction):
    true_label = np.greater(true_label, 0).astype(np.float32)
    prediction = np.greater(prediction, 0).astype(np.float32)
    intersection = np.sum(true_label * prediction)
    dice = (2. * intersection) / (np.sum(true_label) + np.sum(prediction))
    return dice


true_label_img = imageio.imread('')
predicted_img = imageio.imread('')

true_label_img_binary = (true_label_img > 127).astype(np.int32)
predicted_img_binary = (predicted_img > 127).astype(np.int32)

# 计算Dice系数
dice_score = dice_coefficient(true_label_img_binary, predicted_img_binary)
print(f'Dice Score: {dice_score}')