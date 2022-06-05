async function start() {
    // Load the model.
    console.log('started')
    const tfliteModel = await tflite.loadTFLiteModel(
      "./asset/model.tflite",
    );
  
    // Setup the trigger button.
    setupTrigger(tfliteModel);
  }
  
  async function upsample(tfliteModel) {
    // Prepare the input tensors from the image.
    const inputTensor = tf.image
      // Resize.
      .resizeBilinear(tf.browser.fromPixels(document.querySelector("img")), [
        50,
        50
      ])
      // Normalize.
      .expandDims()
      // .div(255)
      // .sub(1);
    
    // Run the inference and get the output tensors.
    const outputTensor = tfliteModel.predict(inputTensor);
    
    // Process and draw the result on the canvas.
    //
    // De-normalize.
    const data = outputTensor;//.add(1).mul(127.5);
    // Convert from RGB to RGBA, and create and return ImageData.
    const rgb = Array.from(data.dataSync());
    const rgba = [];
    for (let i = 0; i < rgb.length / 3; i++) {
      for (let c = 0; c < 3; c++) {
        rgba.push(rgb[i * 3 + c]);
      }
      rgba.push(255);
    }console.log(rgba)
    console.log(rgb.length)
    // Draw on canvas.
    const imageData = new ImageData(Uint8ClampedArray.from(rgba), 200, 200);
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    ctx.putImageData(imageData, 0, 0);
    canvas.classList.remove("hide");
  }
  
  function setupTrigger(tfliteModel) {
    const trigger = document.querySelector(".trigger");
    trigger.textContent = "Upscale!";
    document.querySelector(".trigger").addEventListener("click", (e) => {
      trigger.textContent = "Processing...";
      setTimeout(() => {
        upsample(tfliteModel);
        trigger.classList.add("hide");
      });
    });

    const trigger1 = document.querySelector(".trigger1");
    trigger1.textContent = "Upscale!";
    document.querySelector(".trigger1").addEventListener("click", (e) => {
      trigger1.textContent = "Processing...";
      setTimeout(() => {
        upsample(tfliteModel);
        trigger.classList.add("hide");
        trigger1.textContent = "Upscale!";
      });
    });
  }

  imgInp.onchange = evt => {
    const [file] = imgInp.files
    if (file) {
      blah.src = URL.createObjectURL(file)
    }
  }
  
  start();
  