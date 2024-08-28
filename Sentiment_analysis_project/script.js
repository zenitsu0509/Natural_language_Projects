document.getElementById('reviewForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const review = document.getElementById('reviewInput').value;
    console.log(review)

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review: review }),
    })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            document.getElementById('ratingOutput').innerText = data.rating;
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

