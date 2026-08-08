import os
import tempfile
import unittest
from timm.data.readers.reader_image_folder import find_images_and_targets


class TestReaderImageFolder(unittest.TestCase):

    def test_root_images_with_class_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy images directly in root directory (no subfolders)
            img1 = os.path.join(temp_dir, "test1.jpg")
            img2 = os.path.join(temp_dir, "test2.png")
            with open(img1, "wb") as f:
                f.write(b"dummy")
            with open(img2, "wb") as f:
                f.write(b"dummy")

            # Provided class map (e.g. from class_map.txt)
            class_to_idx = {"cat": 0, "dog": 1}

            images_and_targets, returned_map = find_images_and_targets(
                temp_dir,
                class_to_idx=class_to_idx,
            )

            # Both images should be found with target None (unlabeled)
            self.assertEqual(len(images_and_targets), 2)
            filepaths = [item[0] for item in images_and_targets]
            targets = [item[1] for item in images_and_targets]

            self.assertIn(img1, filepaths)
            self.assertIn(img2, filepaths)
            self.assertEqual(targets, [None, None])

    def test_unmapped_subfolder_with_class_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subfolder = os.path.join(temp_dir, "unlabeled")
            os.makedirs(subfolder)
            img1 = os.path.join(subfolder, "test1.jpg")
            with open(img1, "wb") as f:
                f.write(b"dummy")

            class_to_idx = {"cat": 0, "dog": 1}

            images_and_targets, _ = find_images_and_targets(
                temp_dir,
                class_to_idx=class_to_idx,
            )

            self.assertEqual(len(images_and_targets), 1)
            self.assertEqual(images_and_targets[0][0], img1)
            self.assertIsNone(images_and_targets[0][1])


if __name__ == "__main__":
    unittest.main()
