## How to run repo locally
1. Download repo:
    ```sh
        git clone git@gitlab.com:xganom00/awesome-skin-vision.git
    ```
2. Create virtual enviroment (recommended)
    * General usage:
    ```sh
        python3 -m venv /path/to/new/virtual/environment
    ```
    * Concrete example:
   ```sh 
        python3 -m venv ./venv  
    ```
3. Activate virtual enviroment:
    ```sh
        source ./venv/bin/activate
    ```
4. Enter downloaded folder:
    ```sh
        cd awesome-skin-vision
    ```
5. Install all required packages:
     ```sh
        pip install -r requirements.txt
    ```
6. Run flask web-app locally:
    ```sh
        flask run 
    ```

## Set swapfile (1GB)
```
sudo fallocate -l 1G /swapfile
sudo dd if=/dev/zero of=/swapfile bs=1024 count=1048576
chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

