# Program results

_updated 2026-08-06 15:36 EDT_

26 completed, 1 failed, 0 running, 0 not started

Chance is 0.50 on M2 and M4. Random-vector control scored M2 0.5017 / M4 0.4932, so anything near those is noise.

| model | hypothesis | M1 PR | M2 broad | M2 strict | M3@10 | M4 AUC | M5 freq |
|---|---|---|---|---|---|---|---|
| cooc | H1,H5 | 115.1 | 0.8677 | 0.9205 | 0.1427 | 0.8432 | 0.443 |
| chem | H1,H5 | 2.7 | 0.5784 | 0.6398 | 0.0349 | 0.5198 | 0.394 |
| core-ii0 | H1,H5 | 41.6 | 0.6585 | 0.7044 | 0.0472 | 0.5157 | 0.455 |
| core-ii0.1 | H1,H5 | 84.7 | 0.8531 | 0.9129 | 0.1365 | 0.7768 | 0.400 |
| core-ii1 | H1,H5 | 109.2 | 0.8708 | 0.9244 | 0.1521 | 0.8278 | 0.409 |
| core-ii10 | H1,H5 | 117.2 | 0.8675 | 0.9226 | 0.1449 | 0.8408 | 0.402 |
| core-ii100 | H1,H5 | 118.8 | 0.8690 | 0.9224 | 0.1407 | 0.8374 | 0.412 |
| svd-ppmi | H2 | 222.7 | 0.7649 | 0.8713 | 0.1108 | 0.7410 | 0.280 |
| glove | H2 | 12.9 | 0.8332 | 0.8918 | 0.1201 | 0.7795 | 0.933 |
| chem-svd | H2 | 140.4 | 0.5827 | 0.6487 | 0.0363 | 0.5132 | 0.386 |
| cooc-s1 | H1,var | 117.2 | 0.8677 | 0.9215 | 0.1437 | 0.8397 | 0.448 |
| core-ii10-s1 | H1,var | 117.7 | 0.8662 | 0.9226 | 0.1464 | 0.8380 | 0.390 |
| chem-s1 | H1,var | 2.7 | 0.5906 | 0.6554 | 0.0371 | 0.5042 | 0.397 |
| cooc-s2 | H1,var | 117.0 | 0.8726 | 0.9220 | 0.1490 | 0.8442 | 0.435 |
| core-ii10-s2 | H1,var | 116.9 | 0.8713 | 0.9228 | 0.1445 | 0.8412 | 0.390 |
| chem-s2 | H1,var | 2.7 | 0.5827 | 0.6478 | 0.0359 | 0.5059 | 0.394 |
| cooc-d32 | H7 | 15.9 | 0.8521 | 0.9073 | 0.1205 | 0.8414 | 0.592 |
| cooc-d64 | H7 | 28.7 | 0.8610 | 0.9152 | 0.1336 | 0.8388 | 0.493 |
| cooc-d128 | H7 | 56.4 | 0.8672 | 0.9199 | 0.1463 | 0.8418 | 0.440 |
| cooc-d600-b16k | H7 | 202.3 | 0.8586 | 0.9140 | 0.1262 | 0.8314 | 0.323 |
| cooc-d300-b16k | H7,control | 126.3 | 0.8636 | 0.9179 | 0.1375 | 0.8388 | 0.334 |
| cooc-full | H8 | 146.1 | 0.8607 | 0.9017 | 0.0971 | 0.7618 | 0.693 |
| core-ii10-full | H8 | 147.8 | 0.8636 | 0.9040 | 0.1038 | 0.7635 | 0.732 |
| cooc-recipeholdout | backtest | 110.6 | 0.8692 | 0.9241 | 0.1549 | 0.8570 | 0.434 |

## H3: after removing the top 3 principal directions

| model | M1 PR | M2 broad | M4 AUC | M5 freq |
|---|---|---|---|---|
| cooc | 170.9 | 0.7620 | 0.6908 | 0.578 |
| chem | 124.7 | 0.5399 | 0.5118 | 0.090 |
| core-ii0 | 103.8 | 0.5740 | 0.5052 | 0.081 |
| core-ii0.1 | 128.7 | 0.7542 | 0.6454 | 0.522 |
| core-ii1 | 164.9 | 0.7743 | 0.6776 | 0.560 |
| core-ii10 | 172.4 | 0.7697 | 0.6907 | 0.608 |
| core-ii100 | 172.0 | 0.7676 | 0.6893 | 0.593 |
| svd-ppmi | 246.0 | 0.7048 | 0.6060 | 0.265 |
| glove | 22.0 | 0.7185 | 0.6823 | 0.099 |
| chem-svd | 182.7 | 0.4892 | 0.5013 | 0.102 |
| cooc-s1 | 170.8 | 0.7604 | 0.6908 | 0.557 |
| core-ii10-s1 | 172.6 | 0.7718 | 0.6856 | 0.598 |
| chem-s1 | 123.3 | 0.5420 | 0.5059 | 0.074 |
| cooc-s2 | 170.5 | 0.7656 | 0.6949 | 0.581 |
| core-ii10-s2 | 171.6 | 0.7714 | 0.6878 | 0.605 |
| chem-s2 | 124.7 | 0.5533 | 0.5043 | 0.079 |
| cooc-d32 | 21.6 | 0.7580 | 0.7011 | 0.436 |
| cooc-d64 | 41.4 | 0.7642 | 0.7237 | 0.447 |
| cooc-d128 | 81.3 | 0.7593 | 0.7292 | 0.415 |
| cooc-d600-b16k | 281.7 | 0.7526 | 0.6729 | 0.670 |
| cooc-d300-b16k | 181.0 | 0.7621 | 0.6795 | 0.667 |
| cooc-full | 184.6 | 0.7235 | 0.6794 | 0.195 |
| core-ii10-full | 186.9 | 0.7278 | 0.6737 | 0.158 |
| cooc-recipeholdout | 162.9 | 0.7743 | 0.7221 | 0.595 |

## cuisines

```
{
  "cuisines": {
    "north_american": {
      "region": "North America",
      "recipes_total": 2669991,
      "recipes_used": 29820,
      "real": 29.940941495881066,
      "null_mean": 28.70813894058156,
      "null_sd": 0.09082774940508528,
      "delta": 1.2328025552995072,
      "ci95": 0.17802238883396715,
      "z": 13.572972614363653,
      "rel_delta": 0.04294261490969827,
      "significant": true,
      "direction": "pairing"
    },
    "chinese": {
      "region": "East Asia",
      "recipes_total": 1433167,
      "recipes_used": 29164,
      "real": 15.529154730612705,
      "null_mean": 15.400750699200584,
      "null_sd": 0.08726120127984578,
      "delta": 0.12840403141212064,
      "ci95": 0.17103195450849773,
      "z": 1.4714905310589321,
      "rel_delta": 0.008337517691185391,
      "significant": false,
      "direction": "pairing"
    },
    "russian": {
      "region": "Eastern Europe",
      "recipes_total": 162762,
      "recipes_used": 29917,
      "real": 25.084770246867002,
      "null_mean": 25.14836129583585,
      "null_sd": 0.08817422446165288,
      "delta": -0.06359104896884915,
      "ci95": 0.17282147994483965,
      "z": -0.7211977123371809,
      "rel_delta": -0.002528635890855392,
      "significant": false,
      "direction": "contrast"
    },
    "turkish": {
      "region": "Middle East",
      "recipes_total": 125748,
      "recipes_used": 29904,
      "real": 21.845598046335564,
      "null_mean": 22.520075908944385,
      "null_sd": 0.07315390772426292,
      "delta": -0.6744778626088213,
      "ci95": 0.1433816591395553,
      "z": -9.219984052678535,
      "rel_delta": -0.029950070565301082,
      "significant": true,
      "direction": "contrast"
    },
    "spanish": {
      "region": "Southern Europe",
      "recipes_total": 46446,
      "recipes_used": 29570,
      "real": 37.16197764760393,
      "null_mean": 35.75930696977226,
      "null_sd": 0.14952881014724992,
      "delta": 1.4026706778316722,
      "ci95": 0.29307646788860986,
      "z": 9.380604824250115,
      "rel_delta": 0.03922533171622608,
      "significant": true,
      "direction": "pairing"
    },
    "vietnamese": {
      "region": "Southeast Asia",
      "recipes_total": 31884,
      "recipes_used": 29400,
      "real": 20.76483120709756,
      "null_mean": 21.21232423111417,
      "null_sd": 0.06858374513560174,
      "delta": -0.44749302401660884,
      "ci95": 0.1344241404657794,
      "z": -6.524767977191082,
      "rel_delta": -0.02109589779700932,
      "significant": true,
      "direction": "contrast"
    },
    "indian": {
      "region": "South Asia",
      "recipes_total": 22204,
      "recipes_used": 22160,
      "real": 36.4258239958878,
      "null_mean": 35.198317256207346,
      "null_sd": 0.11460114778201695,
      "delta": 1.2275067396804502,
      "ci95": 0.22461824965275323,
      "z": 10.711120817177965,
      "rel_delta": 0.03487401771924126,
      "significant": true,
      "direction": "pairing"
    },
    "indonesian": {
      "region": "Southeast Asia",
      "recipes_total": 14939,
      "recipes_used": 14929,
      "real": 23.116729544060945,
      "null_mean": 21.87111173468857,
      "null_sd": 0.1205044199460326,
      "delta": 1.2456178093723764,
      "ci95": 0.2361886630942239,
      "z": 10.336698105598293,
      "rel_delta": 0.05695265172079801,
      "significant": true,
      "direction": "pairing"
    },
    "israeli": {
      "region": "Middle East",
      "recipes_total": 8698,
      "recipes_used": 8586,
      "real": 36.67419312591574,
      "null_mean": 36.847669758012806,
      "null_sd": 0.22605085170078298,
      "delta": -0.17347663209706354,
      "ci95": 0.44305966933353463,
      "z": -0.7674230412840451,
      "rel_delta": -0.004707940372792223,
      "significant": false,
      "direction": "contrast"
    },
    "persian": {
      "region": "Middle East",
      "recipes_total": 5793,
      "recipes_used": 5737,
      "real": 48.02194563377477,
      "n
```

## cuisines-full

```
{
  "cuisines": {
    "north_american": {
      "region": "North America",
      "recipes_total": 2669991,
      "recipes_used": 198903,
      "real": 29.891588265697518,
      "null_mean": 28.65429546176335,
      "null_sd": 0.04339311144167453,
      "delta": 1.2372928039341673,
      "ci95": 0.08505049842568208,
      "z": 28.513576529243245,
      "rel_delta": 0.043180011373346316,
      "significant": true,
      "direction": "pairing"
    },
    "chinese": {
      "region": "East Asia",
      "recipes_total": 1433167,
      "recipes_used": 194438,
      "real": 15.615018340720695,
      "null_mean": 15.40248655617297,
      "null_sd": 0.03304201222348395,
      "delta": 0.2125317845477248,
      "ci95": 0.06476234395802855,
      "z": 6.4321683289213265,
      "rel_delta": 0.013798537253879105,
      "significant": true,
      "direction": "pairing"
    },
    "russian": {
      "region": "Eastern Europe",
      "recipes_total": 162762,
      "recipes_used": 162302,
      "real": 25.279190262459746,
      "null_mean": 25.293182100011677,
      "null_sd": 0.04280714682045513,
      "delta": -0.013991837551930786,
      "ci95": 0.08390200776809206,
      "z": -0.3268575130834198,
      "rel_delta": -0.0005531861312113957,
      "significant": false,
      "direction": "contrast"
    },
    "turkish": {
      "region": "Middle East",
      "recipes_total": 125748,
      "recipes_used": 125323,
      "real": 22.036070261912787,
      "null_mean": 22.688460352704574,
      "null_sd": 0.04929490514434635,
      "delta": -0.652390090791787,
      "ci95": 0.09661801408291884,
      "z": -13.234432420174965,
      "rel_delta": -0.02875426893892423,
      "significant": true,
      "direction": "contrast"
    },
    "spanish": {
      "region": "Southern Europe",
      "recipes_total": 46446,
      "recipes_used": 45803,
      "real": 36.9884897298803,
      "null_mean": 35.635119934128056,
      "null_sd": 0.07425879800523096,
      "delta": 1.3533697957522435,
      "ci95": 0.14554724409025266,
      "z": 18.22504312090951,
      "rel_delta": 0.037978539100021656,
      "significant": true,
      "direction": "pairing"
    },
    "vietnamese": {
      "region": "Southeast Asia",
      "recipes_total": 31884,
      "recipes_used": 31241,
      "real": 20.764377760718826,
      "null_mean": 21.227744862388786,
      "null_sd": 0.0989355723564888,
      "delta": -0.4633671016699594,
      "ci95": 0.19391372181871805,
      "z": -4.683523738057891,
      "rel_delta": -0.021828371533282886,
      "significant": true,
      "direction": "contrast"
    },
    "indian": {
      "region": "South Asia",
      "recipes_total": 22204,
      "recipes_used": 22160,
      "real": 36.4258239958878,
      "null_mean": 35.15724422446349,
      "null_sd": 0.12911853831138848,
      "delta": 1.2685797714243066,
      "ci95": 0.2530723350903214,
      "z": 9.824923578091774,
      "rel_delta": 0.03608302639777408,
      "significant": true,
      "direction": "pairing"
    },
    "indonesian": {
      "region": "Southeast Asia",
      "recipes_total": 14939,
      "recipes_used": 14929,
      "real": 23.116729544060945,
      "null_mean": 21.89116428148287,
      "null_sd": 0.0935420511513101,
      "delta": 1.2255652625780762,
      "ci95": 0.1833424202565678,
      "z": 13.101757418122551,
      "rel_delta": 0.05598447148901751,
      "significant": true,
      "direction": "pairing"
    },
    "israeli": {
      "region": "Middle East",
      "recipes_total": 8698,
      "recipes_used": 8586,
      "real": 36.67419312591574,
      "null_mean": 36.63600405994974,
      "null_sd": 0.21432104513613517,
      "delta": 0.03818906596600158,
      "ci95": 0.4200692484668249,
      "z": 0.1781862623044982,
      "rel_delta": 0.0010423916839705134,
      "significant": false,
      "direction": "pairing"
    },
    "persian": {
      "region": "Middle East",
      "recipes_total": 5793,
      "recipes_used": 5737,
      "real": 48.02194563377477,
      "n
```

## Failed

- cooc-d600
