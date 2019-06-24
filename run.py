from bullseye import App


IMAGE_NAME = 'target.jpg'


if __name__ == '__main__':
	app = App(IMAGE_NAME)
	app.main_loop()
