#/bin/bash -e
# TechNexion Camera Driver Modules Installation for
# Nvidia Jetson ORIN NANO Development Kits.
# Before installing, You should backup your image to avoid
# there are any files you lost. 

SERVICE_FILE="tn_cam.service"
TGT_SERVICE_FILE="/etc/systemd/system/"$SERVICE_FILE

CONF_FILE="/boot/firmware/config.txt"

get_kernel_version() {
    RPI_KERNEL_VER=$(uname -r)
}

install_tncam_module() {
    echo "Install TN-CAM modules: $1"
    TGT_MODULE_FILE=/lib/modules/$RPI_KERNEL_VER/kernel/drivers/media/i2c/$1
    sudo cp $1 $TGT_MODULE_FILE
    echo "Installed TN-CAM module file Done."
}

install_tncam_dtbo() {
    echo "Install TN-CAM DTBO file: $1"
    TGT_DTB_FILE="/boot/firmware/overlays/"$1
    sudo cp $1 $TGT_DTB_FILE
    echo "Installed TN-CAM DTBO file Done."
}

add_tncam_conf() {
    echo "Add TN-CAM Configuration for modules: $1"
	sudo sed -i '/dtoverlay='$1'/d' $CONF_FILE
    sudo sed -i 's/camera_auto_detect=1/camera_auto_detect=0/' $CONF_FILE
    sudo sed -i '/camera_auto_detect=0/ adtoverlay='$1 $CONF_FILE
    if [ -n "$CHECKED_CONF" ]; then
        return 27
    fi
}

install_tncam_service() {
    echo "Install TN-CAM service..."
    SERVICE_CONTENT=$(cat $SERVICE_FILE | \
    sed "s/KERNEL_VER/$RPI_KERNEL_VER/g")
    sudo bash -c 'echo -e "\n\n'"$SERVICE_CONTENT"'" > '"$TGT_SERVICE_FILE"
    echo "Launch TN-CAM Service..."
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_FILE
    sudo systemctl start $SERVICE_FILE 2>/dev/null
}

# Start to install TechNexion Camera Driver for RPI5
sudo echo "" > /dev/null # Access sudo Permissions
cat << EOF
****** TechNexion Camera Driver Installation ******
This installation is easy to install TechnNexion Camera Drivers for Raspberry Pi 5. 
Before start to install camera driver, You should BACKUP your image and config 
to avoid any file you lost while installing process.
EOF
read -p "Do you want to continue?[Y/n]" choice


if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo "Continuing with the installation..."
    get_kernel_version
    install_tncam_module "tevs.ko.xz"
    install_tncam_dtbo "tevs-rpi22.dtbo"

	module="tevs-rpi22"
	add_tncam_conf $module
	module="tevs-rpi22,cam0"
	add_tncam_conf $module
    install_tncam_service

    echo "Finish Camera Driver Installation. Return Code:["$?"]"
    if [ $? -eq 0 ]; then
        echo "You should Reboot Device to enable TEVS Cameras."
        read -p "Do you want to reboot now?[Y/n]" choice
        if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
            echo "Rebooting...."
            sudo reboot
        else
            echo "Reboot manually later."
        fi
    fi
else
    echo "Installation aborted."
fi