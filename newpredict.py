import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas
from Img_converter import nimage
import pygame
genres = ["Rock", "Pop", "Classical", "Hip Hop", "Rythm and blues", "Country", "Jazz", "Electronic"]
genres = [i + " music" for i in genres]
data = np.load("Data.npy", allow_pickle = True)
df = pandas.DataFrame(data)
df.columns = ["data", "tags"]
model = keras.models.load_model('Models')

which = int(input("input img or draw"))
match which:
    case 1:
        pygame.init()

        width, height = 800, 800
        screen = pygame.display.set_mode((width, height))

        pygame.display.set_caption("Drawing Program")

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        running = False

            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (255, 0, 0), mouse_pos, 2)
            pygame.display.update()


        pixels = pygame.surfarray.array3d(screen)
        print(np.unique(pixels))
        resized_pixels = pygame.transform.scale(pixels, (128, 128))

        pygame.quit()
    case 2:
        image = nimage(input("path to img")).x
    case other:
        raise AssertionError("sawee")

image2 = image[None,:,:,:]
ls = model(image2)
predicted_index = tf.argmax(ls, axis=1)
predicted_labels = [df["tags"].to_list()[i] for i in predicted_index]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image)
ax.set_title("Predicted: " + genres[predicted_labels[0]])

plt.tight_layout()
plt.show()
