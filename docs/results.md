# Results

CSV files containing an ImageNet-1K validation and OOD test set validation results for all included models with pretrained weights and default configurations is located [here](https://github.com/rwightman/pytorch-image-models/tree/master/results).

## Self-trained Weights
I've leveraged the training scripts in this repository to train a few of the models with to good levels of performance.

|Model | Acc@1 (Err) | Acc@5 (Err) | Param # (M) | Interpolation | Image Size |
|---|---|---|---|---|---|
| efficientnet_b3a | 81.874 (18.126) | 95.840 (4.160) | 12.23 | bicubic | 320 (1.0 crop) |
| efficientnet_b3 | 81.498 (18.502) | 95.718 (4.282) | 12.23 | bicubic | 300 |
| skresnext50d_32x4d | 81.278 (18.722) | 95.366 (4.634) | 27.5 | bicubic | 288 (1.0 crop) |
| efficientnet_b2a | 80.608 (19.392) | 95.310 (4.690) | 9.11 | bicubic | 288 (1.0 crop) |
| mixnet_xl | 80.478 (19.522) | 94.932 (5.068) | 11.90 | bicubic | 224 |
| efficientnet_b2 | 80.402 (19.598) | 95.076 (4.924) | 9.11 | bicubic | 260 |
| skresnext50d_32x4d | 80.156 (19.844) | 94.642 (5.358) | 27.5 | bicubic | 224 |
| resnext50_32x4d | 79.762 (20.238) | 94.600 (5.400) | 25 | bicubic | 224 |
| resnext50d_32x4d | 79.674 (20.326) | 94.868 (5.132) | 25.1 | bicubic | 224 |
| ese_vovnet39b | 79.320 (20.680) | 94.710 (5.290) | 24.6 | bicubic | 224 |
| resnetblur50 | 79.290 (20.710) | 94.632 (5.368) | 25.6 | bicubic | 224 |
| resnet50 | 79.038 (20.962) | 94.390 (5.610) | 25.6 | bicubic | 224 |
| mixnet_l | 78.976 (21.024 | 94.184 (5.816) | 7.33 | bicubic | 224 |
| efficientnet_b1 | 78.692 (21.308) | 94.086 (5.914) | 7.79 | bicubic | 240 |
| efficientnet_es | 78.066 (21.934) | 93.926 (6.074) | 5.44 | bicubic | 224 |
| seresnext26t_32x4d | 77.998 (22.002) | 93.708 (6.292) | 16.8 | bicubic | 224 |
| seresnext26tn_32x4d | 77.986 (22.014) | 93.746 (6.254) | 16.8 | bicubic | 224 |
| efficientnet_b0 | 77.698 (22.302) | 93.532 (6.468) | 5.29 | bicubic | 224 |
| seresnext26d_32x4d | 77.602 (22.398) | 93.608 (6.392) | 16.8 | bicubic | 224 |
| mobilenetv2_120d | 77.294 (22.706 | 93.502 (6.498) | 5.8 | bicubic | 224 |
| mixnet_m | 77.256 (22.744) | 93.418 (6.582) | 5.01 | bicubic | 224 |
| seresnext26_32x4d | 77.104 (22.896) | 93.316 (6.684) | 16.8 | bicubic | 224 |
| skresnet34 | 76.912 (23.088) | 93.322 (6.678) | 22.2 | bicubic | 224 |
| ese_vovnet19b_dw | 76.798 (23.202) | 93.268 (6.732) | 6.5 | bicubic | 224 |
| resnet26d | 76.68 (23.32) | 93.166 (6.834) | 16 | bicubic | 224 |
| densenetblur121d | 76.576 (23.424) | 93.190 (6.810) | 8.0 | bicubic | 224 |
| mobilenetv2_140 | 76.524 (23.476) | 92.990 (7.010) | 6.1 | bicubic | 224 |
| mixnet_s | 75.988 (24.012) | 92.794 (7.206) | 4.13 | bicubic | 224 |
| mobilenetv3_large_100 | 75.766 (24.234) | 92.542 (7.458) | 5.5 | bicubic | 224 |
| mobilenetv3_rw | 75.634 (24.366) | 92.708 (7.292) | 5.5 | bicubic | 224 |
| mnasnet_a1 | 75.448 (24.552) | 92.604 (7.396) | 3.89 | bicubic | 224 |
| resnet26 | 75.292 (24.708) | 92.57 (7.43) | 16 | bicubic | 224 |
| fbnetc_100 | 75.124 (24.876) | 92.386 (7.614) | 5.6 | bilinear | 224 |
| resnet34 | 75.110 (24.890) | 92.284 (7.716) | 22 | bilinear | 224 |
| mobilenetv2_110d | 75.052 (24.948) | 92.180 (7.820) | 4.5 | bicubic | 224 |
| seresnet34 | 74.808 (25.192) | 92.124 (7.876) | 22 | bilinear | 224 |
| mnasnet_b1 | 74.658 (25.342) | 92.114 (7.886) | 4.38 | bicubic | 224 |
| spnasnet_100 | 74.084 (25.916)  | 91.818 (8.182) | 4.42 | bilinear | 224 |
| skresnet18 | 73.038 (26.962) | 91.168 (8.832) | 11.9 | bicubic | 224 |
| mobilenetv2_100 | 72.978 (27.022) | 91.016 (8.984) | 3.5 | bicubic | 224 |
| seresnet18 | 71.742 (28.258) | 90.334 (9.666) | 11.8 | bicubic | 224 |

## Ported Weights
For the models below, the model code and weight porting from Tensorflow or MXNet Gluon to Pytorch was done by myself. There are weights/models ported by others included in this repository, they are not listed below.

| Model | Acc@1 (Err) | Acc@5 (Err) | Param # (M) | Interpolation | Image Size |
|---|---|---|---|---|---|
| tf_efficientnet_l2_ns *tfp | 88.352 (11.648) | 98.652 (1.348) | 480 | bicubic | 800 |
| tf_efficientnet_l2_ns      | TBD | TBD | 480 | bicubic | 800 |
| tf_efficientnet_l2_ns_475      | 88.234 (11.766) | 98.546 (1.454)f | 480 | bicubic | 475 |
| tf_efficientnet_l2_ns_475 *tfp | 88.172 (11.828) | 98.566 (1.434) | 480 | bicubic | 475 |
| tf_efficientnet_b7_ns *tfp | 86.844 (13.156) | 98.084 (1.916) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7_ns      | 86.840 (13.160) | 98.094 (1.906) | 66.35 | bicubic | 600 |
| tf_efficientnet_b6_ns      | 86.452 (13.548) | 97.882 (2.118) | 43.04 | bicubic | 528 |
| tf_efficientnet_b6_ns *tfp | 86.444 (13.556) | 97.880 (2.120) | 43.04 | bicubic | 528 |
| tf_efficientnet_b5_ns *tfp | 86.064 (13.936) | 97.746 (2.254) | 30.39 | bicubic | 456 |
| tf_efficientnet_b5_ns      | 86.088 (13.912) | 97.752 (2.248) | 30.39 | bicubic | 456 |
| tf_efficientnet_b8_ap *tfp | 85.436 (14.564) | 97.272 (2.728) | 87.4 | bicubic | 672 |
| tf_efficientnet_b8 *tfp    | 85.384 (14.616) | 97.394 (2.606) | 87.4 | bicubic | 672 |
| tf_efficientnet_b8         | 85.370 (14.630) | 97.390 (2.610) | 87.4 | bicubic | 672 |
| tf_efficientnet_b8_ap      | 85.368 (14.632) | 97.294 (2.706) | 87.4 | bicubic | 672 |
| tf_efficientnet_b4_ns *tfp | 85.298 (14.702) | 97.504 (2.496) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4_ns      | 85.162 (14.838) | 97.470 (2.530) | 19.34 | bicubic | 380 |
| tf_efficientnet_b7_ap *tfp | 85.154 (14.846) | 97.244 (2.756) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7_ap      | 85.118 (14.882) | 97.252 (2.748) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7 *tfp    | 84.940 (15.060) | 97.214 (2.786) | 66.35 | bicubic | 600 |
| tf_efficientnet_b7         | 84.932 (15.068) | 97.208 (2.792) | 66.35 | bicubic | 600 |
| tf_efficientnet_b6_ap      | 84.786 (15.214) | 97.138 (2.862) | 43.04 | bicubic | 528 |
| tf_efficientnet_b6_ap *tfp | 84.760 (15.240) | 97.124 (2.876) | 43.04 | bicubic | 528 |
| tf_efficientnet_b5_ap *tfp | 84.276 (15.724) | 96.932 (3.068) | 30.39 | bicubic | 456 |
| tf_efficientnet_b5_ap      | 84.254 (15.746) | 96.976 (3.024) | 30.39 | bicubic | 456 |
| tf_efficientnet_b6 *tfp    | 84.140 (15.860) | 96.852 (3.148) | 43.04 | bicubic | 528 |
| tf_efficientnet_b6         | 84.110 (15.890) | 96.886 (3.114) | 43.04 | bicubic | 528 |
| tf_efficientnet_b3_ns *tfp | 84.054 (15.946) | 96.918 (3.082) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3_ns      | 84.048 (15.952) | 96.910 (3.090) | 12.23 | bicubic | 300 |
| tf_efficientnet_b5 *tfp    | 83.822 (16.178) | 96.756 (3.244) | 30.39 | bicubic | 456 |
| tf_efficientnet_b5         | 83.812 (16.188) | 96.748 (3.252) | 30.39 | bicubic | 456 |
| tf_efficientnet_b4_ap *tfp | 83.278 (16.722) | 96.376 (3.624) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4_ap      | 83.248 (16.752) | 96.388 (3.612) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4         | 83.022 (16.978) | 96.300 (3.700) | 19.34 | bicubic | 380 |
| tf_efficientnet_b4 *tfp    | 82.948 (17.052) | 96.308 (3.692) | 19.34 | bicubic | 380 |
| tf_efficientnet_b2_ns *tfp | 82.436 (17.564) | 96.268 (3.732) | 9.11 | bicubic | 260 |
| tf_efficientnet_b2_ns      | 82.380 (17.620) | 96.248 (3.752) | 9.11 | bicubic | 260 |
| tf_efficientnet_b3_ap *tfp | 81.882 (18.118) | 95.662 (4.338) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3_ap      | 81.828 (18.172) | 95.624 (4.376) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3         | 81.636 (18.364) | 95.718 (4.282) | 12.23 | bicubic | 300 |
| tf_efficientnet_b3 *tfp    | 81.576 (18.424) | 95.662 (4.338) | 12.23 | bicubic | 300 |
| tf_efficientnet_lite4      | 81.528 (18.472) | 95.668 (4.332) | 13.00  | bilinear | 380 |
| tf_efficientnet_b1_ns *tfp | 81.514 (18.486) | 95.776 (4.224) | 7.79 | bicubic | 240 |
| tf_efficientnet_lite4 *tfp | 81.502 (18.498) | 95.676 (4.324) | 13.00  | bilinear | 380 |
| tf_efficientnet_b1_ns      | 81.388 (18.612) | 95.738 (4.262) | 7.79 | bicubic | 240 |
| gluon_senet154           | 81.224 (18.776) | 95.356 (4.644) | 115.09 | bicubic | 224 |
| gluon_resnet152_v1s      | 81.012 (18.988) | 95.416 (4.584) | 60.32  | bicubic | 224 |
| gluon_seresnext101_32x4d | 80.902 (19.098) | 95.294 (4.706) | 48.96  | bicubic | 224 |
| gluon_seresnext101_64x4d | 80.890 (19.110) | 95.304 (4.696) | 88.23  | bicubic | 224 |
| gluon_resnext101_64x4d   | 80.602 (19.398) | 94.994 (5.006) | 83.46  | bicubic | 224 |
| tf_efficientnet_el       | 80.534 (19.466) | 95.190 (4.810) | 10.59 | bicubic | 300 |
| tf_efficientnet_el *tfp  | 80.476 (19.524) | 95.200 (4.800) | 10.59 | bicubic | 300 |
| gluon_resnet152_v1d      | 80.470 (19.530) | 95.206 (4.794) | 60.21  | bicubic | 224 |
| gluon_resnet101_v1d      | 80.424 (19.576) | 95.020 (4.980) | 44.57  | bicubic | 224 |
| tf_efficientnet_b2_ap *tfp | 80.420 (19.580) | 95.040 (4.960) | 9.11 | bicubic | 260 |
| gluon_resnext101_32x4d   | 80.334 (19.666) | 94.926 (5.074) | 44.18  | bicubic | 224 |
| tf_efficientnet_b2_ap    | 80.306 (19.694) | 95.028 (4.972) | 9.11 | bicubic | 260 |
| gluon_resnet101_v1s      | 80.300 (19.700) | 95.150 (4.850) | 44.67  | bicubic | 224 |
| tf_efficientnet_b2 *tfp  | 80.188 (19.812) | 94.974 (5.026) | 9.11  | bicubic | 260 |
| tf_efficientnet_b2       | 80.086 (19.914) | 94.908 (5.092) | 9.11  | bicubic | 260 |
| gluon_resnet152_v1c      | 79.916 (20.084) | 94.842 (5.158) | 60.21  | bicubic | 224 |
| gluon_seresnext50_32x4d  | 79.912 (20.088) | 94.818 (5.182) | 27.56  | bicubic | 224 |
| tf_efficientnet_lite3       | 79.812 (20.188) | 94.914 (5.086) | 8.20  | bilinear | 300 |
| tf_efficientnet_lite3 *tfp  | 79.734 (20.266) | 94.838 (5.162) | 8.20  | bilinear | 300 |
| gluon_resnet152_v1b      | 79.692 (20.308) | 94.738 (5.262) | 60.19  | bicubic | 224 |
| gluon_xception65         | 79.604 (20.396) | 94.748 (5.252) | 39.92  | bicubic | 299 |
| gluon_resnet101_v1c      | 79.544 (20.456) | 94.586 (5.414) | 44.57  | bicubic | 224 |
| tf_efficientnet_b1_ap *tfp | 79.532 (20.468) | 94.378 (5.622) | 7.79 | bicubic | 240 |
| tf_efficientnet_cc_b1_8e *tfp | 79.464 (20.536)| 94.492 (5.508) | 39.7 | bicubic | 240 |
| gluon_resnext50_32x4d    | 79.356 (20.644) | 94.424 (5.576) | 25.03  | bicubic | 224 |
| gluon_resnet101_v1b      | 79.304 (20.696) | 94.524 (5.476) | 44.55  | bicubic | 224 |
| tf_efficientnet_cc_b1_8e | 79.298 (20.702) | 94.364 (5.636) | 39.7 | bicubic | 240 |
| tf_efficientnet_b1_ap    | 79.278 (20.722) | 94.308 (5.692) | 7.79 | bicubic | 240 |
| tf_efficientnet_b1 *tfp  | 79.172 (20.828) | 94.450 (5.550) | 7.79  | bicubic | 240 |
| gluon_resnet50_v1d       | 79.074 (20.926) | 94.476 (5.524) | 25.58  | bicubic | 224 |
| tf_efficientnet_em *tfp  | 78.958 (21.042) | 94.458 (5.542) | 6.90 | bicubic | 240 |
| tf_mixnet_l *tfp         | 78.846 (21.154) | 94.212 (5.788) | 7.33  | bilinear | 224 |
| tf_efficientnet_b1       | 78.826 (21.174) | 94.198 (5.802) | 7.79  | bicubic | 240 |
| tf_efficientnet_b0_ns *tfp | 78.806 (21.194) | 94.496 (5.504) | 5.29 | bicubic | 224 |
| gluon_inception_v3       | 78.804 (21.196) | 94.380 (5.620) | 27.16M | bicubic | 299 |
| tf_mixnet_l              | 78.770 (21.230) | 94.004 (5.996) | 7.33  | bicubic | 224 |
| tf_efficientnet_em       | 78.742 (21.258) | 94.332 (5.668) | 6.90 | bicubic | 240 |
| gluon_resnet50_v1s       | 78.712 (21.288) | 94.242 (5.758) | 25.68  | bicubic | 224 |
| tf_efficientnet_b0_ns    | 78.658 (21.342) | 94.376 (5.624) | 5.29 | bicubic | 224 |
| tf_efficientnet_cc_b0_8e *tfp | 78.314 (21.686) | 93.790 (6.210) | 24.0 | bicubic | 224 |
| gluon_resnet50_v1c       | 78.010 (21.990) | 93.988 (6.012) | 25.58  | bicubic | 224 |
| tf_efficientnet_cc_b0_8e | 77.908 (22.092) | 93.656 (6.344) | 24.0 | bicubic | 224 |
| tf_inception_v3          | 77.856 (22.144) | 93.644 (6.356) | 27.16M | bicubic | 299 |
| tf_efficientnet_cc_b0_4e *tfp | 77.746 (22.254) | 93.552 (6.448) | 13.3 | bicubic | 224 |
| tf_efficientnet_es *tfp  | 77.616 (22.384) | 93.750 (6.250) | 5.44 | bicubic | 224 |
| gluon_resnet50_v1b       | 77.578 (22.422) | 93.718 (6.282) | 25.56  | bicubic | 224 |
| adv_inception_v3         | 77.576 (22.424) | 93.724 (6.276) | 27.16M | bicubic | 299 |
| tf_efficientnet_lite2 *tfp  | 77.544 (22.456) | 93.800 (6.200) | 6.09  | bilinear | 260 |
| tf_efficientnet_lite2       | 77.460 (22.540) | 93.746 (6.254) | 6.09  | bicubic | 260 |
| tf_efficientnet_b0_ap *tfp | 77.514 (22.486) | 93.576 (6.424) | 5.29  | bicubic | 224 |
| tf_efficientnet_cc_b0_4e | 77.304 (22.696) | 93.332 (6.668) | 13.3 | bicubic | 224 |
| tf_efficientnet_es       | 77.264 (22.736) | 93.600 (6.400) | 5.44 | bicubic | 224 |
| tf_efficientnet_b0 *tfp  | 77.258 (22.742) | 93.478 (6.522) | 5.29  | bicubic | 224 |
| tf_efficientnet_b0_ap    | 77.084 (22.916) | 93.254 (6.746) | 5.29  | bicubic | 224 |
| tf_mixnet_m *tfp         | 77.072 (22.928) | 93.368 (6.632) | 5.01  | bilinear | 224 |
| tf_mixnet_m              | 76.950 (23.050) | 93.156 (6.844) | 5.01  | bicubic | 224 |
| tf_efficientnet_b0       | 76.848 (23.152) | 93.228 (6.772) | 5.29  | bicubic | 224 |
| tf_efficientnet_lite1 *tfp  | 76.764 (23.236) | 93.326 (6.674) | 5.42  | bilinear | 240 |
| tf_efficientnet_lite1       | 76.638 (23.362) | 93.232 (6.768) | 5.42  | bicubic | 240 |
| tf_mixnet_s *tfp         | 75.800 (24.200) | 92.788 (7.212) | 4.13  | bilinear | 224 |
| tf_mobilenetv3_large_100 *tfp | 75.768 (24.232) | 92.710 (7.290) | 5.48 | bilinear | 224 |
| tf_mixnet_s              | 75.648 (24.352) | 92.636 (7.364) | 4.13  | bicubic | 224 |
| tf_mobilenetv3_large_100 | 75.516 (24.484) | 92.600 (7.400) | 5.48 | bilinear | 224 |
| tf_efficientnet_lite0 *tfp  | 75.074 (24.926) | 92.314 (7.686) | 4.65  | bilinear | 224 |
| tf_efficientnet_lite0       | 74.842 (25.158) | 92.170 (7.830) | 4.65  | bicubic | 224 |
| gluon_resnet34_v1b       | 74.580 (25.420) | 91.988 (8.012) | 21.80 | bicubic | 224 |
| tf_mobilenetv3_large_075 *tfp | 73.730 (26.270) | 91.616 (8.384) | 3.99 | bilinear | 224 |
| tf_mobilenetv3_large_075 | 73.442 (26.558) | 91.352 (8.648) | 3.99 | bilinear | 224 |
| tf_mobilenetv3_large_minimal_100 *tfp | 72.678 (27.322) | 90.860 (9.140) | 3.92 | bilinear | 224 |
| tf_mobilenetv3_large_minimal_100 | 72.244 (27.756) | 90.636 (9.364) | 3.92 | bilinear | 224 |
| tf_mobilenetv3_small_100 *tfp | 67.918 (32.082) | 87.958 (12.042 | 2.54 | bilinear | 224 |
| tf_mobilenetv3_small_100 | 67.918 (32.082) | 87.662 (12.338) | 2.54 | bilinear | 224 |
| tf_mobilenetv3_small_075 *tfp | 66.142 (33.858) | 86.498 (13.502) | 2.04 | bilinear | 224 |
| tf_mobilenetv3_small_075 | 65.718 (34.282) | 86.136 (13.864) | 2.04 | bilinear | 224 |
| tf_mobilenetv3_small_minimal_100 *tfp | 63.378 (36.622) | 84.802 (15.198) | 2.04 | bilinear | 224 |
| tf_mobilenetv3_small_minimal_100 | 62.898 (37.102) | 84.230 (15.770) | 2.04 | bilinear | 224 |

Models with `*tfp` next to them were scored with `--tf-preprocessing` flag. 

The `tf_efficientnet`, `tf_mixnet` models require an equivalent for 'SAME' padding as their arch results in asymmetric padding. I've added this in the model creation wrapper, but it does come with a performance penalty. 

Sources for original weights:
* `tf_efficientnet*`: [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
* `tf_efficientnet_e*`: [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu)
* `tf_mixnet*`: [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)
* `tf_inception*`: [Tensorflow Slim](https://github.com/tensorflow/models/tree/master/research/slim)
* `gluon_*`: [MxNet Gluon](https://gluon-cv.mxnet.io/model_zoo/classification.html)
