mkdir overlays
https://drive.google.com/open?id=1bhrZi8UOq3qopIfklFqcJZRX_X9YIJtu
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bhrZi8UOq3qopIfklFqcJZRX_X9YIJtu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bhrZi8UOq3qopIfklFqcJZRX_X9YIJtu" -O mask_overlays.tar.gz && rm -rf /tmp/cookies.txt
tar -zxvf mask_overlays.tar.gz --directory overlays
rm mask_overlays.tar.gz
