namespace HousePricePrediction
{
    public static class DoubleExtensions
    {
        public static string Output10(this double value)
        {
            var text = value.ToString("F3");
            return $"{text,10}";
        }
    }
}
