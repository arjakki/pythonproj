from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    bio_data = {
        'name': 'Your Name',
        'title': 'Software Developer',
        'bio': 'Welcome to my personal bio page.',
        'skills': ['Python', 'JavaScript', 'Web Development'],
        'email': 'your.email@example.com',
        'github': 'https://github.com/yourusername'
    }
    return render_template('index.html', bio=bio_data)

if __name__ == '__main__':
    app.run(debug=True)