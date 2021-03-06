import os
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import tensorflow as tf
from .utils import (
    VGG_LOSS,
    get_esargan_model,
    preprocess_image,
    downscale_image,
    plt_save_image,
    get_esargan_model,
    get_intermediate,
)


def home_page_view(request):
    return render(request, "home_page.html")


def client_index_view(request):
    return render(request, "client_index.html")


def server_index_view(request):
    form = ImageUploadForm()
    context = {"form": form, "is_processed": False}

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            image = form.cleaned_data.get("upload_image")
            normal_image = "static/img/image.jpg"
            low_img = "static/img/lr_image.jpg"
            sr_img = "static/img/super_resolution.jpg"

            print("Cleaning previous images ...")
            if os.path.exists(normal_image):
                os.remove(normal_image)

            if os.path.exists(low_img):
                os.remove(low_img)

            if os.path.exists(sr_img):
                os.remove(sr_img)

            path = default_storage.save(normal_image, ContentFile(image.read()))
            print(f"Path: {path} type: {type(path)}")

            hr_image = preprocess_image(path)
            print("Saving HR Image")
            plt_save_image(tf.squeeze(hr_image), filename=normal_image)
            print("Saving LR Image")
            lr_image = downscale_image(tf.squeeze(hr_image))
            plt_save_image(tf.squeeze(lr_image), filename=low_img)

            esargan_model = get_esargan_model()
            fake_image = esargan_model(lr_image)
            fake_image = tf.squeeze(fake_image)
            print("Saving Super Resolution Image")
            plt_save_image(tf.squeeze(fake_image), filename=sr_img)
            psnr = tf.image.psnr(
                tf.clip_by_value(fake_image, 0, 255),
                tf.clip_by_value(hr_image, 0, 255),
                max_val=255,
            )

            layer_names = get_intermediate(path)

            context["is_processed"] = True
            context["form"] = form
            context["psnr"] = float(psnr)
            context["layer_names"] = layer_names

    return render(request, "server_index.html", context=context)
