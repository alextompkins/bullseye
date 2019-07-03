import math


class HoughBundler:
	"""Group and merge each cluster of cv2.HoughLinesP() output"""

	# Parameters to play with
	MAX_DISTANCE_TO_MERGE = 10
	MAX_ANGLE_TO_MERGE = 15

	def checker(self, line_new, groups):
		"""Check if line have enough distance and angle to be count as similar"""
		for group in groups:
			# walk through existing line groups
			for line_old in group:
				# check distance
				if self.get_distance(line_old, line_new) < self.MAX_DISTANCE_TO_MERGE:
					# check the angle between lines
					orientation_new = self.get_orientation(line_new)
					orientation_old = self.get_orientation(line_old)
					# if all is ok -- line is similar to others in group
					if abs(orientation_new - orientation_old) < self.MAX_ANGLE_TO_MERGE:
						group.append(line_new)
						return False
		# if it is totally different line
		return True

	@staticmethod
	def get_orientation(line):
		"""get orientation of a line"""
		orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
		return math.degrees(orientation)

	@staticmethod
	def distance_from_line(point, line):
		"""Get distance between point and line"""
		px, py = point
		x1, y1, x2, y2 = line

		def line_magnitude(x1, y1, x2, y2):
			"""Get line (aka vector) length"""
			magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
			return magnitude

		line_mag = line_magnitude(x1, y1, x2, y2)
		if line_mag < 0.00000001:
			distance = 9999
			return distance

		u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
		u = u1 / (line_mag * line_mag)

		if (u < 0.00001) or (u > 1):
			# closest point does not fall within the line segment, take the shorter distance to an endpoint
			ix = line_magnitude(px, py, x1, y1)
			iy = line_magnitude(px, py, x2, y2)
			if ix > iy:
				distance = iy
			else:
				distance = ix
		else:
			# Intersecting point is on the line, use the formula
			ix = x1 + u * (x2 - x1)
			iy = y1 + u * (y2 - y1)
			distance = line_magnitude(px, py, ix, iy)

		return distance

	def get_distance(self, a_line, b_line):
		"""Get all possible distances between both ends of two lines and return the shortest"""
		dist1 = self.distance_from_line(a_line[:2], b_line)
		dist2 = self.distance_from_line(a_line[2:], b_line)
		dist3 = self.distance_from_line(b_line[:2], a_line)
		dist4 = self.distance_from_line(b_line[2:], a_line)

		return min(dist1, dist2, dist3, dist4)

	def group_lines(self, lines):
		"""Group lines by distance"""
		# first line will create new group every time
		groups = [[lines[0]]]
		# if line is different from existing gropus, create a new group
		for line_new in lines[1:]:
			if self.checker(line_new, groups):
				groups.append([line_new])

		return groups

	def merge_line_segments(self, lines):
		"""Sort lines cluster and return first and last coordinates"""
		orientation = self.get_orientation(lines[0])

		# special case
		if len(lines) == 1:
			return [lines[0][:2], lines[0][2:]]

		# [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
		points = []
		for line in lines:
			points.append(line[:2])
			points.append(line[2:])
		# if vertical
		if 45 < orientation < 135:
			# sort by y
			points = sorted(points, key=lambda point: point[1])
		else:
			# sort by x
			points = sorted(points, key=lambda point: point[0])

		# return first and last point in sorted group
		# [[x,y],[x,y]]
		return [points[0], points[-1]]

	def process_lines(self, lines):
		"""Main function for lines from cv.HoughLinesP() output merging
		:param lines cv.HoughLinesP() output"""
		lines_x = []
		lines_y = []
		# for every line of cv2.HoughLinesP()
		for line_i in [l[0] for l in lines]:
			orientation = self.get_orientation(line_i)
			# if vertical
			if 45 < orientation < 135:
				lines_y.append(line_i)
			else:
				lines_x.append(line_i)

		lines_y = sorted(lines_y, key=lambda line: line[1])
		lines_x = sorted(lines_x, key=lambda line: line[0])
		merged_lines_all = []

		# for each cluster in vertical and horizontal lines leave only one line
		for lines in [lines_x, lines_y]:
			if len(lines) > 0:
				groups = self.group_lines(lines)
				merged_lines = []
				for group in groups:
					merged_lines.append(self.merge_line_segments(group))

				merged_lines_all.extend(merged_lines)

		return merged_lines_all
