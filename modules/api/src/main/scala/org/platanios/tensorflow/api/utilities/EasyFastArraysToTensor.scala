package org.platanios.tensorflow.api.utilities

import java.nio.ByteBuffer

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.tensors.Tensor

object FastArraysToTensor {

  /**
    * This TC helps map between types and byte arrays.
    * It also keeps track of byte array sizes for various types.
    * todo: implement ByteSize typeclass for all tf types.
    */
  trait ByteSize[T] {
    def byteSize: Int
    def toBytes(in: T): Array[Byte]
  }
  object ByteSize {
    implicit val FloatByteSize: ByteSize[Float] = new ByteSize[Float] {
      override def byteSize: Int = 4

      override def toBytes(in: Float): Array[Byte] =
        java.nio.ByteBuffer.allocate(4).putFloat(in).array()
    }
  }

  /**
    * This TC will help us determine whether a type is a nested array.
    * @tparam T the type of the nested array. eg., Array[Array[Float]]
    */
  trait NestedArray[T] {
    type DataType
  }
  object NestedArray extends NestedArrayLP {
    trait Aux[T, InnerData] extends NestedArray[T] {
      type DataType = InnerData
    }
    implicit def trivial[T] = new Aux[Array[T], T] {}
  }
  trait NestedArrayLP {
    implicit def complex[T, InnerData](
        implicit na: NestedArray.Aux[T, InnerData]) =
      new NestedArray.Aux[Array[T], InnerData] {}
  }

  /**
    * The main tc for traversing a nested array, getting its tf byte size and shape.
    */
  trait BytesWithShape[T] {
    def bytesWithShape(in: T): (Array[Byte], Array[Int])
  }
  object BytesWithShape {
    implicit def bytesWithShape[T](implicit b: ByteSize[T]) = {
      new BytesWithShape[T] {
        override def bytesWithShape(in: T): (Array[Byte], Array[Int]) =
          (b.toBytes(in).reverse, Array()) //endianness
      }
    }

    implicit def nestedBytesWithShape[T](implicit bws: BytesWithShape[T]) =
      new BytesWithShape[Array[T]] {
        override def bytesWithShape(in: Array[T]): (Array[Byte], Array[Int]) = {
          val all = in.map(bws.bytesWithShape)
          //we require all internal shapes are identical to stack them.
          //todo: NAT support for compile time check.
          require(
            all.forall(_._2.sameElements(all.head._2)),
            "illegal array to tensor. peer arrays must be of same dim: " + all
              .map(_._2)
              .mkString(","))
          all.flatMap(_._1) -> (Array(in.length) ++ all.head._2)
        }
      }
  }

  /**
    * Helper method to get tensors from nested array structures.
    */
  object EasyByteDump {
    def toTensor[T, DataType](in: T)(implicit byteSize: ByteSize[DataType],
                                     na: NestedArray.Aux[T, DataType],
                                     bytesWithShape: BytesWithShape[T],
                                     ev: TF[DataType]) = {
      val bws = bytesWithShape.bytesWithShape(in)
      Tensor.fromBuffer[DataType](Shape(bws._2),
                                  bws._2.product * byteSize.byteSize,
                                  ByteBuffer.wrap(bws._1))
    }
  }
}
