"""
>>  There 3 Nets in MT-CNN (P-Net R-Net O-Net).
>>  There 2 data sets for training(WiderFace for box regression,
    CelebA for key-point regression).

---------------------------------------------------
training phase:

    step 1. 1)  Prepare data set for training P-Net.

                a.  Crop faces from CelebA( resize to 12*12) and adjust the key-point(position).
                b.  Random shift crop faces from WiderFace ( compute IOU with gt one,
                    negative: IOU<0.3, positive: IOU>=0.65, part: 0.4<=IOU<0.65 ), resize to 12*12

            2)  Train P-Net for 10 epochs.

    step 2. 1)  Prepare data set for training R-Net.

                a.  Crop faces from CelebA( resize to 24*24) and adjust the key-point(position).
                b.  Predict face box(candidate) on WiderFace using P-Net ( compute IOU with gt one,
                    negative: IOU<0.3, positive: IOU>=0.65, part: 0.4<=IOU<0.65 ), resize to 24*24.

            2)  Train R-Net for 10 epochs.

    step 3. 1)  Prepare data set for training O-Net.

                a.  Crop faces from CelebA( resize to 48*48) and adjust the key-point(position).
                b.  Predict face box(candidate) on WiderFace using R-Net ( compute IOU with gt one,
                    negative: IOU<0.3, positive: IOU>=0.65, part: 0.4<=IOU<0.65 ), resize to 48*48.

            2)  Train R-Net for 10 epochs.

Note:

    When training:
        we just use small size cropped image(12*12, 24*24, 48*48),
        target(location) is the offset position

    While testing/predict,
        we use big size original image(h*w), just one image.
        scale it to many size(make sure its size >= 12 )...
        feed these images to P-Net, get candidate box(scale on original image).
        feed cropped images to R-Net, get offset, adjust candidate box.
        feed cropped-cropped images to O-Net, get offset, adjust candidate box.

    Actually, predict of O-Net is our need result.

    Data set used for training MT-CNN is dynamic.(random crop or cropped by x-Net)

---------------------------------------------------
"""
