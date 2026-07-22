# Portfolio Content Manager

A one-page Streamlit app for editing every JSON file behind your portfolio
site, with a tab per file (Home, Skills, Projects, Education, Experience,
CSR, Beyond Work) and real image uploads — you never type an image path,
you upload the file and the app saves it into the right folder and writes
the path into the JSON for you.

## 1. Place the files

Drop `app.py` (and `requirements.txt`) into the **root of your portfolio
project**, next to your existing `JSON_Files/` folder, so the layout looks
like this:

```
your-portfolio-project/
  app.py                 <- from this download
  JSON_Files/
    home.json
    skills.json
    projects.json
    education.json
    experience.json
    csr-images.json
    beyondwork-images.json
  Images/
  Carousel1/
  CSR/
  BeyondWork/
    ART/
    Badminton/
    ...
```

If `Images/`, `Carousel1/`, `CSR/`, or `BeyondWork/<category>/` don't exist
yet, the app creates them automatically the first time you upload into them.

## 2. Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## 3. Editing

- Each tab across the top matches one JSON file.
- Text fields edit in place; changes stay in the browser session until you
  press the **Save** button at the bottom of that tab — that's the only
  moment the JSON file on disk is overwritten.
- **Image uploads**: use the file uploader wherever an image is shown. The
  file is written straight into the matching folder (`Carousel1/` for
  certifications, `CSR/` for CSR entries, `BeyondWork/<Category>/` for
  hobby photos, `Images/` for the hero photo), and the JSON's `image` /
  `photo` field is updated to that relative path automatically.
- **Add buttons** (`➕ Add project`, `➕ Add certification`, `➕ Add category`,
  etc.) append a new blank entry to that section — fill it in and hit Save.
- **🗑 Remove** deletes an entry from the list (it does not delete the image
  file from disk, only removes it from the JSON).

## Notes

- Uploading a file with the same name as an existing one overwrites it on
  disk — rename before uploading if you want to keep both.
- The Beyond Work tab lets you rename a category; images already uploaded
  keep their old folder path until you re-upload them, so rename first,
  then re-upload images if you want them under the new folder name.
