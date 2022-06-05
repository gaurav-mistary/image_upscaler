from django import forms

class ImageUploadForm(forms.Form):
    upload_image = forms.ImageField()
    upload_image.widget.attrs.update(
        {
            "class": "btn first",
            "id": "image_input"
        }
    )