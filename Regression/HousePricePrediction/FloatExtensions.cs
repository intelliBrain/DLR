namespace HousePricePrediction
{
    public static class FloatExtensions
    {
        public static float Trim(this float value, int digits)
        {
            var factor = 1.0f;
            switch (digits)
            {
                case 0: factor = 1.0f; break;
                case 1: factor = 10.0f; break;
                case 2: factor = 100.0f; break;
                case 3: factor = 1000.0f; break;
                case 4: factor = 10000.0f; break;
            }

            var intValue =(int)(value * factor);
            return intValue / factor;
        }
    }
}
