set xlabel "wavelength, mkm"
plot [0.4:0.7][] "<paste fE.dat fdip.dat" u (2*pi/$1):((($2*cos($3)*$5*sin($6)-$2*sin($3)*$5*cos($6))*$1/2/($5**2*$1**4/3)+1)) w l lw 3 t "Purcell factor, W/W_0+1, FDTD", \
"~/tmp/AgDrudeAir_z=80_horizontaldipole.dat" u 1:2 w l lw 3 t "Theory"
pause -1
