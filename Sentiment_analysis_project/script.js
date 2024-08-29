document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('reviewForm');
    const reviewInput = document.getElementById('reviewInput');
    const ratingOutput = document.getElementById('ratingOutput');
    const resetButton = document.getElementById('resetButton');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const review = reviewInput.value;

        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: review }),
        });

        const data = await response.json();
        ratingOutput.textContent = data.rating;
    });

    resetButton.addEventListener('click', () => {
        reviewInput.value = '';  // Clear the review input
        ratingOutput.textContent = 'N/A';  // Reset the rating output
    });
});
