from data_handling import DataGenerator


def main():
    images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
    generator = DataGenerator(images_dir, (0, 3, 5), 32)
    x, y = generator[0]
    pass


if __name__ == '__main__':
    main()
