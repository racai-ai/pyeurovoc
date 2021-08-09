import requests
import sys
import os


def download_file(url, save_path):
    download_path = save_path + "_part"

    if os.path.exists(download_path):
        os.remove(download_path)

    with open(download_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}{}] {}% [{:.2f}/{:.2f} MB]'.format(
                    '=' * (done-1), ">",  '.' * (50 - done), done*2,
                    downloaded / (1024 * 1024),
                    total / (1024 * 1024)
                ))
                sys.stdout.flush()
    sys.stdout.write('\n\n')

    os.rename(download_path, save_path)
