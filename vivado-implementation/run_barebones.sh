#!/bin/sh

echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t "

if [[ "$1x" == "x" ]]; then
    echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t usage: ./runbarebone.sh <FPGA package> <algo precision> <reuse factor>"
    exit
fi

SCRIPT_DIR="$( 
  cd "$(dirname "$(readlink "$0" || printf %s "$0")")"
  pwd -P 
)"


echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Script Path = $SCRIPT_DIR"

FPGA=$1
echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t FPGA = $FPGA"
PRECISION=$2
echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t PRECISION = $PRECISION"
REUSE=$3
echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t REUSE = $REUSE"


# setup Vivado
. ~ntran/setup_2017-2.sh #setup vivado and vivado_hls


###################### HLS

echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Preparing for HLS..."

SRC_HLS_DIR=$SCRIPT_DIR/hls4ml/keras-to-hls/keras-config.yml
DEST_HLS_DIR=$SCRIPT_DIR/hls4ml/keras-to-hls/modkeras-config.yml

echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Source = $SRC_HLS_DIR"
echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Destination = $DEST_HLS_DIR"

if true; then #for debugging

	cp $SRC_HLS_DIR $DEST_HLS_DIR

	sed -i s/putDir.*/putDir\:\ my-hls-test/g $DEST_HLS_DIR
	sed -i s/Part.*/Part\:\ $FPGA/g $DEST_HLS_DIR
	sed -i s/ReuseFactor.*/ReuseFactor\:\ $REUSE/g $DEST_HLS_DIR
	sed -i s/Precision.*/Precision\:\ ap_fixed$PRECISION/g $DEST_HLS_DIR
	
	echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Running HLS..."
	echo
	
	cd $SCRIPT_DIR/hls4ml/keras-to-hls
	python keras-to-hls.py -c modkeras-config.yml
	cd my-hls-test
	vivado_hls -f build_prj.tcl

fi

###################### Build vivado project
if true; then #for debugging

	echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Preparing for Vivado project..."

	SRC_BUILD_DIR=$SCRIPT_DIR/build.tcl
	DEST_BUILD_DIR=$SCRIPT_DIR/modbuild.tcl

	echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Source = $SRC_BUILD_DIR"
	echo -e `date +"%h%y %T"` "BB [${LINENO}]  \t Destination = $DEST_BUILD_DIR"

	cp $SRC_BUILD_DIR $DEST_BUILD_DIR

	sed -i s/\-part.*/\-part\ $FPGA/g $DEST_BUILD_DIR
	sed -i s/\"part\".*/\"part\"\ \"$FPGA\"\ \$obj/g $DEST_BUILD_DIR

	ESCAPED_PATH=$(echo "${SCRIPT_DIR}/hls4ml/keras-to-hls/my-hls-test/myproject_prj/solution1/impl/ip/xilinx_com_hls_myproject_1_0.zip" | sed s/\\\//\\\\\\\//g)

	#echo "Escape " $ESCAPED_PATH

	sed -i s/\".*repo_path\ user_ip_repo/\"$ESCAPED_PATH\"\ \-repo_path\ user_ip_repo/g $DEST_BUILD_DIR

	sed -i s/simple_algo_array_hw/myproject_prj_1/g $DEST_BUILD_DIR
	sed -i s/\-name.*\ \-ve/\-name\ myproject\ \-ve/g $DEST_BUILD_DIR
	sed -i s/\-vendor\ \".*\"\ \-lib/\-lib/g $DEST_BUILD_DIR

	ESCAPED_PATH=$(echo "${SCRIPT_DIR}/" | sed s/\\\//\\\\\\\//g)
	sed -i s/\"\$.*top\.vhd/\"${ESCAPED_PATH}barebones_top.vhd/g $DEST_BUILD_DIR
	sed -i s/\"\$.*\.xdc/\"${ESCAPED_PATH}barebones_top.xdc/g $DEST_BUILD_DIR


	cd $SCRIPT_DIR

	./barebones $FPGA $PRECISION

	rm -rf $SCRIPT_DIR/top  $SCRIPT_DIR/user_ip_repo vivado*.jou vivado*.log
	vivado -mode batch -source $DEST_BUILD_DIR

fi




