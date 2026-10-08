### Check need for improvements for VAE (on testing set)
- [x] reconstruction from sampled z
- [x] generation from random z
- [x] generation by changing the label laving the encoded image the same
- [x] assemble complete model:
    - [x] input test images to the MLP classifier
    - [x] use the predicted label to generate the successor
    - [x] input the successor to the decoder, alongside z ~N(0, 1)
    - [x] output the couple (successor label, generated successor image)