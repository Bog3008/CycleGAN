from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
from ModelLightWrapper import GeneratorWrapper
import sys

# console args parsing
common_transf_paths = ''
if len(sys.argv) < 2:
    print('\033[91mThe path to the folder is not specified.\033[0m')
else:
    common_transf_paths = sys.argv[1]
    common_transf_paths = os.path.abspath(common_transf_paths)

### model part
print('preparing the model...')
transformtions_paths = {}
trans_names = ['horse2zebra', 'zebra2horse', 'summer2winter', 'winter2summer']
file_names = ['gen_zebras', 'gen_horse', 'gen_winter', 'gen_summer']
for name, path in zip(trans_names, file_names):
    transformtions_paths[name] = os.path.join(common_transf_paths, path)
    

generator = GeneratorWrapper(transformtions_paths)
#generator.load_model(trans_names[0]) #can be random number [0:len(trans_names))
# if it coinside we dont need to load at app runtime 
print('the model is prepared')
### web app part
app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder


def apply_test_transformation(image_path, transformation):
    img = Image.open(image_path)

    if transformation == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transformation == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif transformation == 'rotate90':
        img = img.rotate(90)
    elif transformation == 'rotate-90':
        img = img.rotate(-90)

    img.save(image_path)
    img.close()



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        img_full_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(img_full_path)

        img = os.path.join(app.config['UPLOAD'], 'server_error.jpg')
        # Check if 'transformation' is in the form data
        if 'transformation' in request.form:
            transformation = request.form['transformation']
            try:
                img = generator(transformation=transformation, image_path=img_full_path)
        
            except Exception as e:
                print(e)
                # var img still contains path to server error img
                #pass
        else:
            # Handle the case where 'transformation' is not in the form data
            # var img still contains path to server error img
            print("Transformation not specified.")

        return render_template('image_render.html', img=img)
    return render_template('image_render.html')


if __name__ == '__main__':
    app.run(debug=True, port=9000)
