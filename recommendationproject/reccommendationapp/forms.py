from django import forms

class UserRecommendationForm(forms.Form):
    user_id = forms.IntegerField(label='User ID', min_value=1)
