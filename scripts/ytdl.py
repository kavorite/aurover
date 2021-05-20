import ctypes
import re
import socket
from io import BytesIO
from os import cpu_count, environ, path, utime

import av
import pythoncom
import win32clipboard
from pytube import Playlist, YouTube
from tqdm import tqdm

socket.setdefaulttimeout(8)


def make_progress_callbacks(num_tracks=None):
    progress = tqdm(unit="track", total=num_tracks)
    totaled = set()

    def on_progress(stream, chunk, remaining):
        if progress.total == None or progress.total < len(totaled):
            totaled.add(stream)
            progress.total = len(totaled)
            progress.refresh()

    def on_complete(stream, file):
        progress.update()

    return on_progress, on_complete


def resolve_metadata(video):
    data_attached = list(video.metadata)
    if data_attached:
        metadata = data_attached[0]
    else:
        keywords = (
            "spotify",
            "music",
            "bandcamp",
            "itunes",
            "soundcloud",
            "beatport",
            "soundtrack",
        )
        song_promo = False
        for meta in (video.description.lower(), video.title.lower()):
            if True in (s in meta for s in keywords):
                song_promo = True
                break
        if song_promo and "-" in video.title:
            artist, title = re.split(r"\s+-\s+", video.title, 1)
        else:
            artist, title = re.sub(r" - Topic$", "", video.author), video.title
        metadata = dict(title=title, artist=artist)
        if video.publish_date:
            metadata["year"] = str(video.publish_date.year)
    return metadata


from concurrent import futures


def download_one(link, on_progress, on_complete):
    video = YouTube(link)
    video.register_on_complete_callback(on_complete)
    video.register_on_progress_callback(on_progress)
    audio = (
        video.streams.filter(only_audio=True, audio_codec="opus")
        .order_by("abr")
        .desc()
        .first()
    )

    metadata = resolve_metadata(video)

    total = audio.filesize_approx

    istrm = BytesIO()
    # update_progress(istrm, b"", audio.filesize_approx)
    audio.stream_to_buffer(istrm)
    istrm.seek(0)
    opath = path.join(environ["USERPROFILE"], "Downloads")
    opath = path.join(opath, f"{video.title}.ogg")

    with av.open(istrm) as icontainer:
        src = icontainer.streams.audio[0]
        with av.open(opath, "wb+") as ocontainer:
            ocontainer.metadata.update(metadata)
            dst = ocontainer.add_stream(template=src)
            dst.metadata.update(metadata)
            for pkt in icontainer.demux(src):
                if pkt.dts is not None:
                    pkt.stream = dst
                    ocontainer.mux(pkt)

    return opath


def download_many(link):
    if re.search(r"[?&]list=", link):
        links = tuple(Playlist(link).video_urls)
        on_progress, on_complete = make_progress_callbacks(len(links))

        def download(track):
            while True:
                try:
                    return download_one(track, on_progress, on_complete)
                except (ConnectionError, TimeoutError, socket.timeout):
                    continue

        with futures.ThreadPoolExecutor() as pool:
            opaths = list(pool.map(download, links))
        for opath in opaths:
            # set access/write times in the order that the files' corresponding
            # URLs appear in the source playlist
            with open(opath, "a"):
                utime(opath)
        return opaths
    else:
        on_progress, on_complete = make_progress_callbacks(1)
        return [download_one(link, on_progress, on_complete)]


def clip_files(files):
    class DROPFILES(ctypes.Structure):
        _fields_ = (
            ("pFiles", ctypes.wintypes.DWORD),
            ("pt", ctypes.wintypes.POINT),
            ("fNC", ctypes.wintypes.BOOL),
            ("fWide", ctypes.wintypes.BOOL),
        )

    if not files:
        return

    offset = ctypes.sizeof(DROPFILES)
    length = sum(len(p) + 1 for p in files) + 1
    size = offset + length * ctypes.sizeof(ctypes.c_wchar)
    buf = (ctypes.c_char * size)()
    df = DROPFILES.from_buffer(buf)
    df.pFiles, df.fWide = offset, True
    for path in files:
        array_t = ctypes.c_wchar * (len(path) + 1)
        path_buf = array_t.from_buffer(buf, offset)
        path_buf.value = path
        offset += ctypes.sizeof(path_buf)
    stg = pythoncom.STGMEDIUM()
    stg.set(pythoncom.TYMED_HGLOBAL, buf)
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    try:
        win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, stg.data)
    finally:
        win32clipboard.CloseClipboard()


def clipboard_content():
    try:
        win32clipboard.OpenClipboard(None)
        return win32clipboard.GetClipboardData()
    finally:
        win32clipboard.CloseClipboard()


def main():
    link = clipboard_content()
    files = download_many(link)
    clip_files(files)


if __name__ == "__main__":
    main()
