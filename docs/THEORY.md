# Theory

## Apical four-chamber view

An apical four-chamber view shows both ventricles and both atria. A cavity mask must identify the left ventricle.

## End diastole and end systole

ED is the filled state. ES is the contracted state.

`EF = (V_ED - V_ES) / V_ED`

`EF_percent = 100 * EF`

`V_ED`: end-diastolic volume [pixel cubed before scale].

`V_ES`: end-systolic volume [pixel cubed before scale].

`EF`: ejection fraction [unitless].

`EF_percent`: ejection fraction [percent].

## Area ratio

`EF_area = 1 - A_ES / A_ED`

`A_ED`: end-diastolic cavity area [pixel squared].

`A_ES`: end-systolic cavity area [pixel squared].

`EF_area`: area-ratio estimate [unitless].

Area is not volume. This estimate is a 2D proxy.

## Area-power volume proxy

`V proportional to A^(3/2)`

`EF_area_power = 1 - (A_ES / A_ED)^(3/2)`

`A`: cavity area [pixel squared].

`V`: proxy volume [pixel cubed up to a constant].

`EF_area_power`: proxy ejection fraction [unitless].

This relation assumes similar scaling in all three directions. A real ventricle may not meet this assumption.

## Method of disks

`V = (pi / 4) * (L / N) * sum(d_i^2)`

`V`: volume [pixel cubed before scale].

`L`: long-axis length [pixel].

`N`: number of disks [20].

`d_i`: diameter of disk `i` [pixel].

The method estimates volume from diameters across the long axis. This code estimates the axis with PCA. It needs a valid cavity mask.

## Dice score

`Dice = 2 * |P intersection G| / (|P| + |G|)`

`P`: predicted mask [pixels].

`G`: reference mask [pixels].

`Dice`: overlap score [0 to 1].

## Speckle noise and denoising

Ultrasound has multiplicative speckle noise. Median and bilateral filters can reduce it. They can also change image features. Denoising is off by default because a model trained on raw frames can fail on filtered frames.
