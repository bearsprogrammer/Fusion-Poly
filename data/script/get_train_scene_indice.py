"""Export official train scene numbers to data/train_indice.json."""
if __package__:
    from ._scene_indices import main
else:
    from _scene_indices import main


if __name__ == '__main__':
    main('train')
