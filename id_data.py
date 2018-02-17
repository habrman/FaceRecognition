import os
import detect_and_align
from scipy import misc
import numpy as np


class ID_Data():
    def __init__(self, name, image_path):
        self.name = name
        self.image_path = image_path
        self.embedding = []


def get_id_data(id_folder, pnet, rnet, onet, sess, embeddings, images_placeholder, phase_train_placeholder):
    id_dataset = []
    ids = os.listdir(os.path.expanduser(id_folder))
    ids.sort()
    for id_name in ids:
        id_dir = os.path.join(id_folder, id_name)
        image_names = os.listdir(id_dir)
        image_paths = [os.path.join(id_dir, img) for img in image_names]
        for image_path in image_paths:
            id_dataset.append(ID_Data(id_name, image_path))

    aligned_images = align_id_dataset(id_dataset, pnet, rnet, onet)

    feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)

    for i in range(len(id_dataset)):
        id_dataset[i].embedding = emb[i, :]
    return id_dataset


def align_id_dataset(id_dataset, pnet, rnet, onet):
    aligned_images = []

    for i in range(len(id_dataset)):
        image = misc.imread(os.path.expanduser(id_dataset[i].image_path), mode='RGB')
        face_patches, _, _ = detect_and_align.align_image(image, pnet, rnet, onet)
        aligned_images = aligned_images + face_patches

    aligned_images = np.stack(aligned_images)
    return aligned_images


if __name__ == "__main__":
    main()
