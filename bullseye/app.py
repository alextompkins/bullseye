from enum import Enum
from math import atan2, degrees

import cv2
import numpy as np

from bullseye.helpers import average_point, centre_of_contour, dist, mean_and_standard_dev, group, avg_of_groups
from bullseye.hough_bundler import HoughBundler


class Colour(Enum):
	"""Definitions of colours in BGR colour space"""
	BLUE = (255, 0, 0)
	GREEN = (0, 255, 0)
	RED = (0, 0, 255)
	YELLOW = (0, 255, 255)
	MAGENTA = (255, 0, 255)
	CYAN = (255, 255, 0)
	TARGET_GOLD = (0, 212, 251)
	TARGET_RED = (25, 25, 217)
	TARGET_BLUE = (206, 158, 0)
	TARGET_BLACK = (20, 19, 19)


class ScoreColourThreshold(Enum):
	"""HSV thresholds for each colour."""
	GOLD = (20, 100, 0), (40, 255, 255)
	RED = (165, 100, 0), (200, 255, 255)
	BLUE = (30, 150, 0), (100, 250, 255)
	BLACK = (0, 0, 0), (255, 35, 150)


class Ellipse:
	def __init__(self, centre, dimensions, angle):
		self.x, self.y = centre
		self.width, self.height = dimensions
		self.angle = angle

	def to_tuple(self):
		return (self.x, self.y), (self.width, self.height), self.angle

	def get_centre(self):
		return self.x, self.y


class App:
	def __init__(self, sample_img_name):
		self.image = cv2.imread('sample_data/{}'.format(sample_img_name))

	def main_loop(self):
		exited = False
		while not exited:
			resized = self.resize_image(self.image)
			gold, red, blue, black = self.segment_regions(resized)

			annotated = resized.copy()
			contours = dict()
			for name, region in ('GOLD', gold), ('RED', red), ('BLUE', blue), ('BLACK', black):
				largest_contour = self.find_biggest_contour(region)
				contours[name] = largest_contour

			for contour in contours.values():
				pass
				# cv2.drawContours(annotated, contour, -1, Colour.MAGENTA.value, thickness=2)
				# cv2.circle(annotated, centre_of_contour(contour), 2, Colour.RED.value, thickness=10)

			centres = tuple(map(lambda c: centre_of_contour(c), contours.values()))
			true_centre = average_point(centres)
			cv2.circle(annotated, tuple(true_centre[0]), 2, Colour.BLUE.value, thickness=5)

			score_regions = dict()  # score regions mapped to their radius
			for colour, contour in contours.items():
				mean_dist, std_dev = mean_and_standard_dev(contour, key=lambda pt: dist(true_centre, pt))
				filtered = tuple(filter(lambda pt: abs(dist(true_centre, pt) - mean_dist) < std_dev, contour))
				# cv2.drawContours(annotated, filtered, -1, Colour.YELLOW.value, thickness=1)
				filtered = np.asarray(filtered)

				# Draw a circle for each region
				mean_dist, std_dev = mean_and_standard_dev(filtered, key=lambda pt: dist(true_centre, pt))
				score_regions[colour] = int(mean_dist)
				cv2.circle(annotated, centre_of_contour(filtered), int(mean_dist), Colour['TARGET_{}'.format(colour)].value, thickness=2)

				# Draw an ellipse for each region
				# ellipse = Ellipse(*cv2.fitEllipse(filtered))
				# cv2.ellipse(annotated, ellipse.to_tuple(), Colour.GREEN.value, thickness=2)

			edges = self.find_edges(resized)

			arrow_impacts = []
			filtered_lines = []
			lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)
			if lines is not None:
				for line in lines:
					for x1, y1, x2, y2 in line:
						closest_to_centre = min((x1, y1), (x2, y2), key=lambda pt: dist(true_centre, [pt]))
						if dist(true_centre, [closest_to_centre]) < score_regions['BLACK']:
							filtered_lines.append(line)
							# cv2.line(annotated, (x1, y1), (x2, y2), Colour.BLUE.value, thickness=2)

			merged_lines = HoughBundler().process_lines(filtered_lines)
			print(len(filtered_lines), len(merged_lines))
			for line in merged_lines:
				(x1, y1), (x2, y2) = line
				closest_to_centre = min((x1, y1), (x2, y2), key=lambda pt: dist(true_centre, [pt]))
				cv2.line(annotated, (x1, y1), (x2, y2), Colour.GREEN.value, thickness=2)
				arrow_impacts.append(closest_to_centre)

			for impact in arrow_impacts:
				# print(impact)
				cv2.circle(annotated, impact, radius=2, color=Colour.CYAN.value, thickness=2)

			cv2.imshow('annotated', annotated)

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

		gold = cv2.inRange(image, *ScoreColourThreshold.GOLD.value)
		gold = cv2.morphologyEx(gold, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		cv2.imshow('gold', gold)

		red = cv2.inRange(image, *ScoreColourThreshold.RED.value)
		red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		cv2.imshow('red', red)

		blue = cv2.inRange(image, *ScoreColourThreshold.BLUE.value)
		blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		cv2.imshow('blue', blue)

		black = cv2.inRange(image, *ScoreColourThreshold.BLACK.value)
		black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, (10, 10), iterations=5)
		cv2.imshow('black', black)

		# combined = cv2.add(cv2.add(cv2.add(gold, red), blue), black)
		# cv2.imshow('combined', combined)

		return gold, red, blue, black

	@staticmethod
	def find_biggest_contour(binary):
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		largest_contour = max(contours, key=cv2.contourArea)
		return largest_contour

	@staticmethod
	def find_edges(image):
		image = image.copy()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(image, (5, 5), 0)
		edges = cv2.Canny(image, 100, 300)

		cv2.imshow('edges', edges)
		return edges

	@staticmethod
	def find_circles(orig, grey):
		annotated_image = orig.copy()

		circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 10, param1=10, param2=75, minRadius=0, maxRadius=0)
		circles = np.uint16(np.around(circles))
		for circle in circles[0]:
			# draw the outer circle
			cv2.circle(annotated_image, (circle[0], circle[1]), circle[2], Colour.GREEN.value, 2)
			# draw the center of the circle
			cv2.circle(annotated_image, (circle[0], circle[1]), 2, Colour.RED.value, 3)

		cv2.imshow('hough_circles', annotated_image)
		return annotated_image


def prune_circle_contour(contour):
	pruned = []
	prev_angle = None
	for i in range(len(contour) - 1):
		pt, next_pt = contour[i][0], contour[i + 1][0]
		dx = next_pt[0] - pt[0]
		dy = next_pt[1] - pt[1]
		angle = degrees(atan2(dy, dx))
		# if abs(angle) > 3:
		print(pt, next_pt, angle)

	return pruned
