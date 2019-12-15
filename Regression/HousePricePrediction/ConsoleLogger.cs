using System;
using System.IO;
using System.Text;
using System.Collections.Generic;

namespace HousePricePrediction
{
    public class ConsoleLogger : IDisposable
    {
        private string outputDir;
        private string outputFilePath;
        private List<string> lines = new List<string>();
        private StringBuilder currentLine = new StringBuilder();

        public string OutputDir => outputDir;

        public ConsoleLogger(string rootPath, string runId)
        {
            var fileInfo = new FileInfo(Path.Combine(rootPath, $"{runId}.txt"));
            outputDir = fileInfo.Directory.FullName;
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            outputFilePath = fileInfo.FullName;

            if (File.Exists(outputFilePath))
            {
                File.Delete(outputFilePath);
            }

            Console.WriteLine($"output: {outputFilePath}");
        }

        public string BuildFilePath(string fileName, bool deleteIfExists)
        {
            var fileInfo = new FileInfo(Path.Combine(outputDir, fileName));
            outputDir = fileInfo.Directory.FullName;

            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            var filePath = fileInfo.FullName;
            if (deleteIfExists)
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }

            return filePath;
        }

        public void Write(string text)
        {
            currentLine.Append(text ?? "");
        }

        public void WriteLine(string text)
        {
            currentLine.Append(text ?? "");

            var finalLine = currentLine.ToString();
            currentLine = new StringBuilder();

            Console.WriteLine(finalLine);
            lines.Add(finalLine);

            if (lines.Count > 10)
            {
                File.AppendAllLines(outputFilePath, lines, Encoding.UTF8);
                lines = new List<string>();
            }
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    if (lines.Count > 0)
                    {
                        File.AppendAllLines(outputFilePath, lines, Encoding.UTF8);
                        lines = new List<string>();
                    }
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        // ~ConsoleLogger()
        // {
        //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        //   Dispose(false);
        // }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion
    }
}
