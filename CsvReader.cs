using System;
using System.Buffers;
using System.Buffers.Text;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipelines;
using System.Threading.Tasks;

namespace NN
{
    public class CsvReader
    {
        public async Task<IEnumerable<DataSet>> Read(string fileName)
        {
            var results = new List<DataSet>();

            using var file = File.OpenRead(fileName);
            var reader = PipeReader.Create(file);

            while (true)
            {
                var result = await reader.ReadAsync();

                var buffer = result.Buffer;
                SequencePosition? position = null;

                do
                {
                    position = buffer.PositionOf((byte)'\n');

                    if (position == null)
                        continue;

                    var row = buffer.Slice(0, position.Value);
                    var dataSet = ReadTrainingSet(row);
                    results.Add(dataSet);

                    buffer = buffer.Slice(buffer.GetPosition(1, position.Value));
                } while (position != null);

                reader.AdvanceTo(buffer.Start, buffer.End);

                if (result.IsCompleted)
                    break;
            }

            reader.Complete();

            return results;
        }

        private DataSet ReadTrainingSet(ReadOnlySequence<byte> row)
        {
            var inputs = new double[784];
            var outputs = new double[10];
            var separator = (byte)',';

            var sequenceReader = new SequenceReader<byte>(row);
            if (!sequenceReader.TryReadTo(out ReadOnlySpan<byte> digitBytes, separator) ||
                !Utf8Parser.TryParse(digitBytes, out int digit, out var _))
                throw new InvalidOperationException("Cannot parse label.");

            outputs[digit] = 1.0;

            for (var inputIndex = 0; !sequenceReader.End; inputIndex++)
            {
                if (sequenceReader.TryReadTo(out ReadOnlySpan<byte> bytes, separator))
                {
                    if (!Utf8Parser.TryParse(bytes, out int pixelValue, out var _))
                        throw new InvalidOperationException("Cannot parse pixel information.");

                    inputs[inputIndex] = AdjustValue(pixelValue);
                }
                else if (!sequenceReader.End)
                {
                    var unreadSpan = sequenceReader.UnreadSpan;
                    if (!Utf8Parser.TryParse(unreadSpan, out int pixelValue, out var _))
                        throw new InvalidOperationException("Cannot parse pixel information.");

                    sequenceReader.Advance(unreadSpan.Length);
                    inputs[inputIndex] = AdjustValue(pixelValue);
                }
                else
                {
                    break;
                }
            }

            return new DataSet(inputs, outputs);
        }

        private double AdjustValue(double value)
        {
            return value / 256.0;
        }
    }
}