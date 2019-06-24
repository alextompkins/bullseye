import cv2


class App:
	def __init__(self, sample_img_name):
		self.image = cv2.imread('sample_data/{}'.format(sample_img_name))

	def main_loop(self):
		exited = False
		while not exited:
			image = self.resize_image(self.image)
			image = self.segment_regions(image)
			cv2.imshow('output', image)

			pressed_key = cv2.waitKey(100) & 0xFF
			if pressed_key == ord('q'):
				exited = True
		cv2.destroyAllWindows()

	@staticmethod
	def resize_image(image):
		return cv2.resize(image, None, fx=0.2, fy=0.2)

	@staticmethod
	def segment_regions(image):
		image = image.copy()
		image = cv2.GaussianBlur(image, (5, 5), 0)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		blue = cv2.inRange(image, (175, 100, 0), (200, 255, 255))
		blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		cv2.imshow('blue', blue)

		# red = cv2.inRange(image, (, 100, 0), (360, 255, 255))
		# red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		# cv2.imshow('red', red)

		yellow = cv2.inRange(image, (20, 100, 0), (40, 255, 255))
		yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		cv2.imshow('yellow', yellow)

		return image
