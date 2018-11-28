from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib


def visualize_heatmap(heatmap, orig, path_for_saving, heatmap_type='binary'):
    # visualizes and saves heatmap on image
    matplotlib.use('Agg')
    fig = plt.figure(frameon=False)
    fig_save = plt.gcf()
    DPI = fig_save.get_dpi()
    fig_save.set_size_inches(64.0 / float(DPI), 64.0 / float(DPI))

    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')

    plt.imshow(heatmap, cmap=heatmap_type)  # sets type of heatmap

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    plt.savefig("/tmp/heatmap_tmp.jpg", bbox_inches=extent)  # saving small heatmap
    plt.close(fig)

    open_orig = Image.open(orig)
    size = open_orig.size
    img = Image.open("/tmp/heatmap_tmp.jpg")
    rsz_img = img.resize(size, Image.ANTIALIAS)
    rsz_img.save("/tmp/heatmap_resized_tmp.jpeg", "jpeg")  # saving resized heatmap
    rsz_img.save(path_for_saving + "map_prediction_for_" + str(orig.rsplit('/', 1)[1]),"jpeg")

    fig = plt.figure(figsize=(17, 9), frameon=False)

    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')

    img = mpimg.imread(str(orig))
    img2 = mpimg.imread("/tmp/heatmap_resized_tmp.jpeg")

    plt.imshow(img)
    plt.imshow(img2, alpha=0.5)

    print ("saving: " + path_for_saving + "prediction_for_" + str(orig.rsplit('/', 1)[1]))

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # saving final visualization
    plt.savefig(path_for_saving + "prediction_for_" + str(orig.rsplit('/', 1)[1]), dpi=120, bbox_inches=extent)
    plt.close(fig)

